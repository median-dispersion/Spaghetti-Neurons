#ifndef LAYER_H
#define LAYER_H

#include <cstddef>
#include <vector>
#include "Neuron.h"
#include <fstream>

class Network;

class Layer {

    public:

        Layer(
            Network* const network,
            const std::size_t neurons
        );

        Layer(
            Network* const network,
            std::ifstream& file
        );

        void connect(Layer* const layer);
        void setActivations(const std::vector<double>& activations);
        void activate();
        std::vector<double> getActivations();
        void setTargets(const std::vector<double>& targets);
        void train();
        void save(std::ofstream& file);

    private:

        Network* const _network;
        std::vector<Neuron*> _neurons;

};

#endif