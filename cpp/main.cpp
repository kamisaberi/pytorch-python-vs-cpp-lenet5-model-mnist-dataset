#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>

#define DEBUG_MODE true


using namespace std;

struct Net : torch::nn::Module {
    Net() {
        conv1 = register_module("conv1", torch::nn::Conv2d(1, 6, /*kernel_size=*/5));
        conv2 = register_module("conv2", torch::nn::Conv2d(6, 16, /*kernel_size=*/5));
        fc1 = register_module("fc1", torch::nn::Linear(16 * 5 * 5, 120));
        fc2 = register_module("fc2", torch::nn::Linear(120, 84));
        fc3 = register_module("fc3", torch::nn::Linear(84, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
//        cout << x.sizes() << endl;
//        cout << "1\n";
        x = torch::relu(conv1->forward(x));
//        cout << x.sizes() << endl;
        x = torch::max_pool2d(x, /*kernel_size=*/2);
//        cout << x.sizes() << endl;
//        cout << "2\n";
        x = torch::relu(conv2->forward(x));
//        cout << "3\n";
        x = torch::max_pool2d(x, /*kernel_size=*/2);
//        cout << x.sizes() << endl;
        x = x.view({-1, 16 * 5 * 5}); // Flatten
//        cout << x.sizes() << endl;
//        cout << "4\n";
        x = torch::relu(fc1->forward(x));
//        cout << "4.1\n";
        x = torch::relu(fc2->forward(x));
//        cout << "5\n";
        x = fc3->forward(x);
        return x;
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};



void set_random()
{
    torch::manual_seed(1);
    torch::cuda::manual_seed_all(1);
    srand(1);
}



// Function to resize a single tensor
torch::Tensor resize_tensor(const torch::Tensor& tensor, const std::vector<int64_t>& size) {
    return torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(false)
    ).squeeze(0);
}

int main() {


    std::cout.precision(10);
    torch::Device device(torch::kCPU);

    // Load the MNIST dataset
    auto dataset = torch::data::datasets::MNIST("/home/kami/datasets/MNIST/raw");

    // Define the target size
    // std::vector<int64_t> size = {32, 32};

    // Create a lambda function for resizing
    auto resize_transform = torch::data::transforms::Lambda<torch::data::Example<>>(
            [](torch::data::Example<> example) {
                example.data = resize_tensor(example.data, {32, 32});
                return example;
            }
    );

    // Apply the resize transform to the dataset
    auto transformed_dataset = dataset.map(resize_transform).map(torch::data::transforms::Normalize<>(0.5,0.5)).map(torch::data::transforms::Stack<>());


    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(transformed_dataset), 64);


    Net model;
    model.to(device);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    for (size_t epoch = 0; epoch != 10; ++epoch) {
        size_t batch_index = 0;

        auto train_loader_interator = train_loader->begin();
        auto train_loader_end = train_loader->end();

        while(train_loader_interator != train_loader_end) {
            torch::Tensor  data,targets;
            auto batch = *train_loader_interator;
            data = batch.data;
            targets = batch.target;
            optimizer.zero_grad();

            torch::Tensor output;
            output = model.forward(data);

            torch::Tensor loss;
            loss = torch::nll_loss(output, targets);
            loss.backward();
            optimizer.step();

            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() << std::endl;
            }
            ++train_loader_interator;

        }
    }




    // Print the size of the original and resized images
//    std::cout << "Original image size: " << train_loader[0].data.sizes() << std::endl;
//    std::cout << "Resized image size: " << transformed_dataset[0].data.sizes() << std::endl;

    return 0;
}
