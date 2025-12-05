#include "Network.h"
#include <stdexcept>
#include <cmath>

// ================================================================================================
// Constructor
// ================================================================================================
Network::Network(
    const std::vector<std::size_t>& topology,
    const double learningRate
):
    _learningRate(learningRate)
{

    for (std::size_t layer = 0; layer < topology.size(); layer++) {

        createLayer(topology[layer]);

        if (layer > 0) {

            Layer* const source = _layers[layer].get();
            Layer* const target = _layers[layer - 1].get();

            source->connect(target);

        }

    }

}

// ================================================================================================
// Construct a network from disk
// ================================================================================================
Network::Network(
    std::ifstream& file,
    const double learningRate
):
    _learningRate(learningRate)
{

    std::size_t layers;
    std::size_t inputs;

    file.read(reinterpret_cast<char*>(&layers), sizeof(layers));
    file.read(reinterpret_cast<char*>(&inputs), sizeof(inputs));

    createLayer(inputs);

    for (std::size_t layer = 1; layer < layers; layer++) {

        loadLayer(file);

    }

}

// ================================================================================================
// Create a new layer in the network
// ================================================================================================
Layer* Network::createLayer(const std::size_t neurons) {

    _layers.emplace_back(std::make_unique<Layer>(this, neurons));

    return _layers.back().get();

}

// ================================================================================================
// Create a new neuron in the network
// ================================================================================================
Neuron* Network::createNeuron() {

    _neurons.emplace_back(std::make_unique<Neuron>(this));

    return _neurons.back().get();

}

// ================================================================================================
// Create a new connection in the network
// ================================================================================================
Connection* Network::createConnection(Neuron* const source, Neuron* const target) {

    _connections.emplace_back(std::make_unique<Connection>(this, source, target));

    return _connections.back().get();

}

// ================================================================================================
// Get the neuron count of the network
// ================================================================================================
std::size_t Network::getNeuronCount() {

    return _neurons.size();

}

// ================================================================================================
// Get a pointer to a neuron
// ================================================================================================
Neuron* Network::getNeuron(const std::size_t id) {

    return _neurons[id].get();

}

// ================================================================================================
// Get the learning rate of the network
// ================================================================================================
double Network::getLearningRate() {

    return _learningRate;

}

// ================================================================================================
// Get the network outputs for a given set of inputs
// ================================================================================================
std::vector<double> Network::getOutputs(const std::vector<double>& inputs) {

    _layers.front()->setActivations(inputs);

    for (std::size_t layer = 1; layer < _layers.size(); layer++) {

        _layers[layer]->activate();

    }

    return _layers.back()->getActivations();

}

// ================================================================================================
// Train the network on a given set of inputs and targets
// ================================================================================================
void Network::train(const std::vector<double>& inputs, const std::vector<double>& targets) {

    std::vector<double> outputs = getOutputs(inputs);

    _layers.back()->setTargets(targets);

    for (std::size_t layer = _layers.size() - 2; layer < _layers.size(); layer--) {

        _layers[layer]->train();

    }

}

// ================================================================================================
// Get the loss of the network for given set of inputs and targets
// ================================================================================================
double Network::getLoss(const std::vector<double>& inputs, const std::vector<double>& targets) {

    std::vector<double> outputs = getOutputs(inputs);

    if (targets.size() != outputs.size()) {

        throw std::invalid_argument("Invalid number of targets!");

    }

    double loss = 0.0;

    for (std::size_t index = 0; index < outputs.size(); index++) {

        loss += std::pow(outputs[index] - targets[index], 2);

    }

    return loss;

}

// ================================================================================================
// Save the network to disk
// ================================================================================================
void Network::save(std::ofstream& file) {

    const std::size_t layers = _layers.size();
    const std::size_t inputs = _layers.front()->getNeuronCount();

    file.write(reinterpret_cast<const char*>(&layers), sizeof(layers));
    file.write(reinterpret_cast<const char*>(&inputs), sizeof(inputs));

    for (std::size_t layer = 1; layer < layers; layer++) {

        _layers[layer]->save(file);

    }

}

// ================================================================================================
// Load a layer from disk
// ================================================================================================
Layer* Network::loadLayer(std::ifstream& file) {

    _layers.emplace_back(std::make_unique<Layer>(this, file));

    return _layers.back().get();

}

// ================================================================================================
// Load a neuron from disk
// ================================================================================================
Neuron* Network::loadNeuron(std::ifstream& file) {

    _neurons.emplace_back(std::make_unique<Neuron>(this, file));

    return _neurons.back().get();

}

// ================================================================================================
// Load a connection from disk
// ================================================================================================
Connection* Network::loadConnection(Neuron* const source, std::ifstream& file) {

    _connections.emplace_back(std::make_unique<Connection>(this, source, file));

    return _connections.back().get();

}