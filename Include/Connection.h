#ifndef CONNECTION_H
#define CONNECTION_H

class Network;
class Neuron;

class Connection {

    public:

        Connection(
            Network* const network,
            Neuron* const source,
            Neuron* const target
        );

        Neuron* getSource();
        Neuron* getTarget();
        double getWeight();
        void setWeight(const double weight);

    private:

        Network* const _network;
        Neuron* const _source;
        Neuron* const _target;
        double _weight;

};

#endif