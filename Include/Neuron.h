#ifndef NEURON_H
#define NEURON_H

#include <cstddef>
#include <vector>
#include "Connection.h"
#include <fstream>

class Network;

class Neuron {

    public:

        Neuron(
            Network* const network
        );

        Neuron(
            Network* const network,
            std::ifstream& file
        );

        void connect(Neuron* const neuron);
        void setActivation(const double activation);
        void activate();
        double getActivation();
        void setTarget(const double target);
        void train();
        std::size_t getID();
        void save(std::ofstream& file);

    private:

        Network* const _network;
        const std::size_t _id;
        double _bias;
        double _activation;
        double _delta;
        std::vector<Connection*> _inputs;
        std::vector<Connection*> _outputs;

};

#endif