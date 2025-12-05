#include "Network.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstddef>

#define NETWORK_BINARY "xor.sn"

int main() {

    std::ifstream binary(NETWORK_BINARY, std::ios::binary);

    Network network(binary, 1.0);

    const std::vector<std::vector<double>> inputs  = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};

    for (auto& input : inputs) {

        const std::vector<double> outputs = network.getOutputs(input);

        for (auto& output : outputs) {

            std::cout << input[0] << " ⊕ " << input[1] << " = " << std::round(output) << " > " << output << std::endl;

        }

    }

    double mse = 0.0;

    for (std::size_t index = 0; index < inputs.size(); index++) {

        mse += network.getLoss(inputs[index], targets[index]);

    }

    mse /= inputs.size();

    std::cout << "Mean squared error: " << mse << std::endl;

    return 0;

}