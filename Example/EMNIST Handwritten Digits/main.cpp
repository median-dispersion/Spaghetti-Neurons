#include <string>
#include <cstddef>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "Network.h"
#include <chrono>
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
// Train the network and periodically log training stats
// ================================================================================================
void trainNetwork(Network& network, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets) {

    /* Messy code! */

    std::chrono::high_resolution_clock::time_point startTimestamp = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point updateTimestamp = startTimestamp;

    for (std::size_t index = 0; index < inputs.size(); index++) {

        network.train(inputs[index], targets[index]);

        std::chrono::duration<double> intervalSeconds = std::chrono::high_resolution_clock::now() - updateTimestamp;

        if (intervalSeconds.count() >= 15) {

            std::size_t meanSquareSamples = 100;
            double meanSquareError = 0.0;

            for (std::size_t sample = 0; sample < meanSquareSamples; sample++) {

                meanSquareError += network.getLoss(inputs[sample], targets[sample]);

            }

            meanSquareError /= meanSquareSamples;
            std::chrono::duration<double> durationSeconds = std::chrono::high_resolution_clock::now() - startTimestamp;
            double iterationsPerSecond = index / durationSeconds.count();
            std::size_t remainingSamples = inputs.size() - index;
            double timeRemainingMinutes = remainingSamples / iterationsPerSecond / 60;
            double completedPercentage = 100.0 / inputs.size() * index;
            timeRemainingMinutes = std::round(timeRemainingMinutes * 100.0) / 100.0;
            completedPercentage = std::round(completedPercentage * 100.0) / 100.0;

            std::cout << "Trained on " << index << " out of " << inputs.size() << " (" << completedPercentage << " %) samples" << std::endl;
            std::cout << "Remaining training time: " << timeRemainingMinutes << " minutes" << std::endl;
            std::cout << "Mean square error over the first " << meanSquareSamples << " samples: " << meanSquareError << std::endl;

            updateTimestamp = std::chrono::high_resolution_clock::now();

        }

    }

}

// ================================================================================================
// Test the network and periodically log training stats
// ================================================================================================
void testNetwork(Network& network, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets) {

    /* Messy code! */

    std::size_t correct = 0;

    std::chrono::high_resolution_clock::time_point startTimestamp = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point updateTimestamp = startTimestamp;

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

        std::chrono::duration<double> intervalSeconds = std::chrono::high_resolution_clock::now() - updateTimestamp;

        if (intervalSeconds.count() >= 15) {

            std::chrono::duration<double> durationSeconds = std::chrono::high_resolution_clock::now() - startTimestamp;
            double iterationsPerSecond = input / durationSeconds.count();
            std::size_t remainingSamples = inputs.size() - input;
            double timeRemainingMinutes = remainingSamples / iterationsPerSecond / 60;
            double completedPercentage = 100.0 / inputs.size() * input;
            timeRemainingMinutes = std::round(timeRemainingMinutes * 100.0) / 100.0;
            completedPercentage = std::round(completedPercentage * 100.0) / 100.0;

            std::cout << "Tested " << input << " out of " << inputs.size() << " (" << completedPercentage << " %) samples" << std::endl;
            std::cout << "Remaining test time: " << timeRemainingMinutes << " minutes" << std::endl;

            updateTimestamp = std::chrono::high_resolution_clock::now();

        }

    }

    double accuracy = 100.0 / inputs.size() * correct;
    accuracy = std::round(accuracy * 100.0) / 100.0;
    std::vector<double> outputs = network.getOutputs(inputs[0]);

    std::cout << "Network accuracy: " << accuracy << " %" << std::endl;
    std::cout << "Output of the first sample:" << std::endl;

    for (std::size_t index = 0; index < outputs.size(); index++) {

        double output = std::round(outputs[index] * 10000.0) / 100.0;

        std::cout << index << " = " <<  output << " %" << std::endl;

    }

}

// ================================================================================================
// Main
// ================================================================================================
int main(int argc, char* argv[]) {

    Arguments arguments = getArguments(argc, argv);

    std::cout << "Loading input files..." << std::endl;

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
                      Network({inputs[0].size(), 128, 64, targets[0].size()}, arguments.rate);

    if (arguments.train) {

        for (std::size_t iteration = 0; iteration < arguments.train; iteration++) {

            std::cout << "Starting network training iteration " << iteration + 1 << " out of " << arguments.train << "..." << std::endl;

            trainNetwork(network, inputs, targets);

            std::cout << "Saving network binary file..." << std::endl;

            std::ofstream outputFile(arguments.network, std::ios::binary);
            network.save(outputFile);

        }

    } else {

        std::cout << "Starting network test..." << std::endl;

        testNetwork(network, inputs, targets);

    }

    return 0;

}