#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <cstddef>
#include <memory>
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"

class Network {

    public:

        Network(
            const std::vector<std::size_t>& topology,
            const double learningRate
        );

        Layer* createLayer(const std::size_t neurons);
        Neuron* createNeuron();
        Connection* createConnection(Neuron* const source, Neuron* const target);
        std::size_t getNeuronCount();
        double getLearningRate();
        std::vector<double> getOutputs(const std::vector<double>& inputs);
        void train(const std::vector<double>& inputs, const std::vector<double>& targets);
        double getLoss(const std::vector<double>& inputs, const std::vector<double>& targets);

    private:

        const double _learningRate;
        std::vector<std::unique_ptr<Layer>> _layers;
        std::vector<std::unique_ptr<Neuron>> _neurons;
        std::vector<std::unique_ptr<Connection>> _connections;

};

#endif