#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <numeric>

struct Net : torch::nn::Module {
    torch::nn::Linear fc1, fc2, fc3;




    Net()
        : fc1(784, 128), // MNIST images are 28x28, which is flattened to 784
          fc2(128, 64),
          fc3(64, 10) { // 10 output classes
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        // init_weights();
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x.view({-1, 784})));
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }
    void init_weights(){
        torch::nn::init::constant_(fc1->bias,0.1);
        torch::nn::init::constant_(fc1->weight,0.5);
        torch::nn::init::constant_(fc2->bias,0.1);
        torch::nn::init::constant_(fc2->weight,0.5);
        torch::nn::init::constant_(fc3->bias,0.1);
        torch::nn::init::constant_(fc3->weight,0.5);
    }



};


void set_random()
{
    torch::manual_seed(1);
//    torch::cuda::manual_seed_all(1);
    srand(1);
}


using Task = std::function<void()>;

void measure_and_execute(Task task, long array_elapsed_time[], int& time_step) {
    auto start_time = std::chrono::high_resolution_clock::now();
    task();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    array_elapsed_time[time_step]+=duration;
    time_step++;
}



int main() {
	std::cout.precision(10);
//    torch::Device device(torch::kCUDA);
    torch::Device device(torch::kCPU);
    set_random();
    auto train_loader =     torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(torch::data::datasets::MNIST("/mnt/hgfs/E/DATASETS/MNIST/raw")
                       .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                       .map(torch::data::transforms::Stack<>())), 64);

    for (auto& batch : *train_loader) {
        auto data = batch.data;
        auto target = batch.target;

        std::cout << "첫 번째 배치의 데이터 텐서 초기값:\n";
        std::cout << data[0][0][10] << std::endl;

        std::cout << "\n첫 번째 배치의 타겟 텐서:\n";
        std::cout << target[0] << std::endl;
        std::cout << target[1] << std::endl;
        std::cout << target[2] << std::endl;
        std::cout << target[3] << std::endl;
        std::cout << target[4] << std::endl;
        std::cout << target[5] << std::endl;

        break;
    }



    set_random();
    Net model;
    model.to(device);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    std::cout << "parameter:" << torch::sum(model.parameters()[0][0])<<std::endl;



    constexpr  int TOTAL_STEP=8;
    long array_elapsed_time[TOTAL_STEP]={0,};
    auto total_epoch_time_start =  std::chrono::high_resolution_clock::now();
    auto total_count = 0;
    model.train();
    for (size_t epoch = 0; epoch != 10; ++epoch) {
        size_t batch_index = 0;

        auto train_loader_interator = train_loader->begin();
        auto train_loader_end = train_loader->end();

        while(train_loader_interator != train_loader_end) {
            auto start_time =  std::chrono::high_resolution_clock::now();
            auto TIME_STEP = 1;

            total_count++;
            torch::Tensor  data,targets;
            auto batch = *train_loader_interator;
            measure_and_execute([&](){ data = batch.data.to(device), targets = batch.target.to(device); }, array_elapsed_time, TIME_STEP);
            measure_and_execute([&](){ optimizer.zero_grad(); }, array_elapsed_time, TIME_STEP);


            torch::Tensor output;
            measure_and_execute([&](){output = model.forward(data); }, array_elapsed_time, TIME_STEP);

            torch::Tensor loss;
            measure_and_execute([&](){ loss = torch::nll_loss(output, targets); }, array_elapsed_time, TIME_STEP);
            measure_and_execute([&](){ loss.backward(); }, array_elapsed_time, TIME_STEP);
            measure_and_execute([&](){ optimizer.step(); }, array_elapsed_time, TIME_STEP);





            if (++batch_index % 100 == 0) {
               measure_and_execute([&](){
                                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                }, array_elapsed_time, TIME_STEP);


            }
            TIME_STEP=0;
            measure_and_execute([&](){ ++train_loader_interator; }, array_elapsed_time, TIME_STEP);


        }
    }


    auto total_epoch_time_end =  std::chrono::high_resolution_clock::now();
    auto total_epoch_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(total_epoch_time_end - total_epoch_time_start).count();


    std::cout << "==================================" << std::endl;
    std::cout << "Total Time: " << total_epoch_elapsed_time << std::endl;
    for (int i = 0; i < TOTAL_STEP; ++i) {
        std::cout << "STEP " << i << ": array_elapsed_time:" << array_elapsed_time[i] << std::endl;
    }

    // std::accumulate를 사용하여 배열의 총합 계산
    long total_elapsed_time = std::accumulate(array_elapsed_time, array_elapsed_time + TOTAL_STEP, 0L);
    std::cout << "Sum Time:" << total_elapsed_time << std::endl;
    std::cout << "Step Count:" << total_count << std::endl;


}

