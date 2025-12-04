#include "Connection.h"
#include "RNG.h"
#include "Network.h"
#include "Neuron.h"

// ================================================================================================
// Constructor
// ================================================================================================
Connection::Connection(
    Network* const network,
    Neuron* const source,
    Neuron* const target
):
    _network(network),
    _source(source),
    _target(target),
    _weight(rng::range(-1.0, 1.0))
{}

// ================================================================================================
// Construct a connection from disk
// ================================================================================================
Connection::Connection(
    Network* const network,
    Neuron* const source,
    std::ifstream& file
):
    _network(network),
    _source(source),
    _target([&network, &file]{ std::size_t id; file.read(reinterpret_cast<char*>(&id), sizeof(id)); return network->getNeuron(id); }()),
    _weight([&file]{ double weight; file.read(reinterpret_cast<char*>(&weight), sizeof(weight)); return weight; }())
{}

// ================================================================================================
// Get a pointer to the source neuron
// ================================================================================================
Neuron* Connection::getSource() {

    return _source;

}

// ================================================================================================
// Get a pointer to the target neuron
// ================================================================================================
Neuron* Connection::getTarget() {

    return _target;

}

// ================================================================================================
// Get the connection weight
// ================================================================================================
double Connection::getWeight() {

    return _weight;

}

// ================================================================================================
// Set the connection weight
// ================================================================================================
void Connection::setWeight(const double weight) {

    _weight = weight;

}

// ================================================================================================
// Save the connection to disk
// ================================================================================================
void Connection::save(std::ofstream& file) {

    const std::size_t target = _target->getID();

    file.write(reinterpret_cast<const char*>(&target), sizeof(target));
    file.write(reinterpret_cast<const char*>(&_weight), sizeof(_weight));

}