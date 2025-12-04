#include <cmath>
#include "Neuron.h"
#include "Network.h"
#include "RNG.h"

static double sigmoid(const double x) { return 1.0 / (1.0 + std::exp(-x)); }
static double sigmoidDerivative(const double x) { return x * (1.0 - x); }

// ================================================================================================
// Constructor
// ================================================================================================
Neuron::Neuron(
    Network* const network
):
    _network(network),
    _id(_network->getNeuronCount()),
    _bias(rng::range(-1.0, 1.0)),
    _activation(0.0),
    _delta(0.0)
{}

// ================================================================================================
// Construct a neuron from disk
// ================================================================================================
Neuron::Neuron(
    Network* const network,
    std::ifstream& file
):
    _network(network),
    _id(_network->getNeuronCount()),
    _bias([&file]{ double bias; file.read(reinterpret_cast<char*>(&bias), sizeof(bias)); return bias; }()),
    _activation(0.0),
    _delta(0.0)
{

    std::size_t connections;

    file.read(reinterpret_cast<char*>(&connections), sizeof(connections));

    for (std::size_t index = 0; index < connections; index++) {

        Connection* const connection = _network->loadConnection(this, file);

        connection->getSource()->_inputs.push_back(connection);
        connection->getTarget()->_outputs.push_back(connection);

    }

}

// ================================================================================================
// Connect this neuron to another neuron
// ================================================================================================
void Neuron::connect(Neuron* const neuron) {

    Connection* const connection = _network->createConnection(this, neuron);

    this->_inputs.push_back(connection);
    neuron->_outputs.push_back(connection);

}

// ================================================================================================
// Set the activation value of the neuron
// ================================================================================================
void Neuron::setActivation(const double activation) {

    _activation = activation;

}

// ================================================================================================
// Activate the neuron
// ================================================================================================
void Neuron::activate() {

    _activation = _bias;

    for (auto& connection : _inputs) {

        _activation += connection->getTarget()->_activation * connection->getWeight();

    }

    _activation = sigmoid(_activation);

}

// ================================================================================================
// Get the activation value of the neuron
// ================================================================================================
double Neuron::getActivation() {

    return _activation;

}

// ================================================================================================
// Set the target value of the neuron
// ================================================================================================
void Neuron::setTarget(const double target) {

    _delta = (target - _activation) * sigmoidDerivative(_activation);

    _bias += _network->getLearningRate() * _delta;

}

// ================================================================================================
// Train the neuron
// ================================================================================================
void Neuron::train() {

    _delta = 0.0;

    for (auto& connection : _outputs) {

        _delta += connection->getSource()->_delta * connection->getWeight();

        connection->setWeight(connection->getWeight() + _network->getLearningRate() * _activation * connection->getSource()->_delta);

    }

    _delta *= sigmoidDerivative(_activation);

    _bias += _network->getLearningRate() * _delta;

}

// ================================================================================================
// Get the neuron ID
// ================================================================================================
std::size_t Neuron::getID() {

    return _id;

}

// ================================================================================================
// Save the neuron to disk
// ================================================================================================
void Neuron::save(std::ofstream& file) {

    const std::size_t connections = _inputs.size();

    file.write(reinterpret_cast<const char*>(&_bias), sizeof(_bias));
    file.write(reinterpret_cast<const char*>(&connections), sizeof(connections));

    for (auto& connection : _inputs) {

        connection->save(file);

    }

}