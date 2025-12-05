#include <string>
#include <cstddef>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "Network.h"
#include <chrono>
#include "RNG.h"
#include <cmath>

struct Arguments {

    std::string network;
    std::string images;
    std::string labels;
    std::size_t train;
    double rate;

};

// ================================================================================================
// Get launch arguments
// ================================================================================================
Arguments getArguments(int argc, char* argv[]) {

    Arguments arguments = {"", "", "", 0, 0.1};

    for (int i = 1; i < argc; i++) {

        std::string argument = argv[i];

        if (argument == "--network") { arguments.network = argv[++i]; }
        if (argument == "--images") { arguments.images = argv[++i]; }
        if (argument == "--labels") { arguments.labels = argv[++i]; }
        if (argument == "--train") { arguments.train = std::stoull(argv[++i]); }
        if (argument == "--rate") { arguments.rate = std::stod(argv[++i]); }

    }

    if (arguments.network.empty() || arguments.images.empty() || arguments.labels.empty()) {

        std::cerr << "Missing input arguments!" << std::endl;
        std::exit(1);

    }

    return arguments;

}

// ================================================================================================
// Get data from file
// ================================================================================================
std::vector<std::vector<double>> getData(std::ifstream& file) {

    std::vector<std::vector<double>> data;

    std::size_t entries;
    std::size_t points;

    file.read(reinterpret_cast<char*>(&entries), sizeof(entries));
    file.read(reinterpret_cast<char*>(&points), sizeof(points));

    for (std::size_t entry = 0; entry < entries; entry++) {

        std::vector<double> values;

        for (std::size_t point = 0; point < points; point++) {

            double value;

            file.read(reinterpret_cast<char*>(&value), sizeof(value));

            values.push_back(value);

        }

        data.push_back(values);

    }

    return data;

}

// ================================================================================================
// Main
// ================================================================================================
int main(int argc, char* argv[]) {

    Arguments arguments = getArguments(argc, argv);

    std::ifstream networkFile(arguments.network, std::ios::binary);
    std::ifstream imagesFile(arguments.images, std::ios::binary);
    std::ifstream labelsFile(arguments.labels, std::ios::binary);

    if (!imagesFile || !labelsFile) {

        std::cerr << "An input file could not be loaded!" << std::endl;
        std::exit(1);

    }

    std::vector<std::vector<double>> inputs = getData(imagesFile);
    std::vector<std::vector<double>> targets = getData(labelsFile);

    if (inputs.size() != targets.size()) {

        std::cerr << "Number of inputs and targets do not match!" << std::endl;
        std::exit(1);

    }

    Network network = networkFile ?
                      Network(networkFile, arguments.rate) :
                      Network({inputs[0].size(), 256, 128, targets[0].size()}, arguments.rate);

    for (std::size_t iteration = 0; iteration < arguments.train; iteration++) {

        auto start = std::chrono::high_resolution_clock::now();

        for (std::size_t index = 0; index < inputs.size(); index++) {

            network.train(inputs[index], targets[index]);

        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Training time: " << duration.count() << " seconds" << std::endl;

        std::size_t index = rng::range(static_cast<std::size_t>(0), inputs.size());
        std::cout << "Mean squared error: " << network.getLoss(inputs[index], targets[index]) << std::endl;

        std::ofstream outputFile(arguments.network, std::ios::binary);
        network.save(outputFile);

    }

    if (!arguments.train) {

        std::size_t correct = 0;

        for (std::size_t input = 0; input < inputs.size(); input++) {

            std::vector<double> outputs = network.getOutputs(inputs[input]);

            std::size_t target = 0;
            std::size_t guess = 0;
            double probability = 0.0;

            for (std::size_t index = 0; index < outputs.size(); index++) {

                if (targets[input][index] == 1) {

                    target = index;

                }

                if (outputs[index] > probability) {

                    guess = index;
                    probability = outputs[index];

                }

            }

            if (guess == target) { correct++; }

        }

        double accuracy = (100.0 / inputs.size()) * correct;
        std::vector<double> outputs = network.getOutputs(inputs[0]);
        std::cout << "Network accuracy: " << accuracy << " %" << std::endl;
        std::cout << "Output of the first input:" << std::endl;

        for (std::size_t index = 0; index < outputs.size(); index++) {

            std::cout << index << " = " << std::round(outputs[index] * 1000.0) / 1000.0 << std::endl;

        }

    }

    return 0;

}