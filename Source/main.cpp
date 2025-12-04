#include "Network.h"
#include <vector>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <fstream>

#define TRAINING_ITERATIONS 100000
#define NETWORK_BINARY "network.sn"

int main() {

    Network network({2, 2, 1}, 1.0);

    const std::vector<std::vector<double>> inputs  = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};

    auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t iteration = 0; iteration < TRAINING_ITERATIONS; iteration++) {

        if (
            iteration == 0 ||
            iteration % (TRAINING_ITERATIONS / 10) == 0 ||
            iteration == TRAINING_ITERATIONS - 1
        ) {

            std::cout << "Iteration " << iteration << " network loss: " << network.getLoss(inputs[0], targets[0]) << std::endl;

        }

        for (std::size_t index = 0; index < inputs.size(); index++) {

            network.train(inputs[index], targets[index]);

        }

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Training time: " << duration.count() << " seconds" << std::endl;

    for (auto& input : inputs) {

        const std::vector<double> outputs = network.getOutputs(input);

        for (auto& output : outputs) {

            std::cout << input[0] << " ⊕ " << input[1] << " = " << output << std::endl;

        }

    }

    std::ofstream output(NETWORK_BINARY, std::ios::binary);
    network.save(output);

    return 0;

}