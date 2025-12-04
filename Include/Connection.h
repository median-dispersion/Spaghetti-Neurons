#ifndef CONNECTION_H
#define CONNECTION_H

#include <fstream>

class Network;
class Neuron;

class Connection {

    public:

        Connection(
            Network* const network,
            Neuron* const source,
            Neuron* const target
        );

        Connection(
            Network* const network,
            Neuron* const source,
            std::ifstream& file
        );

        Neuron* getSource();
        Neuron* getTarget();
        double getWeight();
        void setWeight(const double weight);
        void save(std::ofstream& file);

    private:

        Network* const _network;
        Neuron* const _source;
        Neuron* const _target;
        double _weight;

};

#endif