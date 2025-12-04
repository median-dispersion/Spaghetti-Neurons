#include "Layer.h"
#include "Network.h"
#include <stdexcept>

// ================================================================================================
// Constructor
// ================================================================================================
Layer::Layer(
    Network* const network,
    const std::size_t neurons
):
    _network(network)
{

    for (std::size_t neuron = 0; neuron < neurons; neuron++) {

        _neurons.push_back(_network->createNeuron());

    }

}

// ================================================================================================
// Construct a layer from disk
// ================================================================================================
Layer::Layer(
    Network* const network,
    std::ifstream& file
):
    _network(network)
{

    std::size_t neurons;

    file.read(reinterpret_cast<char*>(&neurons), sizeof(neurons));

    for (std::size_t neuron = 0; neuron < neurons; neuron++) {

        _neurons.push_back(_network->loadNeuron(file));

    }

}

// ================================================================================================
// Connect this layer to another layer
// ================================================================================================
void Layer::connect(Layer* const layer) {

    for (auto& source : _neurons) {

        for (auto& target : layer->_neurons) {

            source->connect(target);

        }

    }

}

// ================================================================================================
// Set the activation values of all neurons in the layer
// ================================================================================================
void Layer::setActivations(const std::vector<double>& activations) {

    if (activations.size() != _neurons.size()) {

        throw std::invalid_argument("Invalid number of activations!");

    }

    for (std::size_t index = 0; index < activations.size(); index++) {

        _neurons[index]->setActivation(activations[index]);

    }

}

// ================================================================================================
// Activate all neurons in the layer
// ================================================================================================
void Layer::activate() {

    for (auto& neuron : _neurons) {

        neuron->activate();

    }

}

// ================================================================================================
// Get the activation values of all neurons in the layer
// ================================================================================================
std::vector<double> Layer::getActivations() {

    std::vector<double> activations;

    for (auto& neuron : _neurons) {

        activations.push_back(neuron->getActivation());

    }

    return activations;

}

// ================================================================================================
// Set the target values of every neuron in this layer
// ================================================================================================
void Layer::setTargets(const std::vector<double>& targets) {

    if (targets.size() != _neurons.size()) {

        throw std::invalid_argument("Invalid number of targets!");

    }

    for (std::size_t index = 0; index < targets.size(); index++) {

        _neurons[index]->setTarget(targets[index]);

    }

}

// ================================================================================================
// Train every neuron in this layer
// ================================================================================================
void Layer::train() {

    for (auto& neuron : _neurons) {

        neuron->train();

    }

}

// ================================================================================================
// Save the layer to disk
// ================================================================================================
void Layer::save(std::ofstream& file) {

    const std::size_t neurons = _neurons.size();

    file.write(reinterpret_cast<const char*>(&neurons), sizeof(neurons));

    for (auto& neuron : _neurons) {

        neuron->save(file);

    }

}