#ifndef NETWORK_H
#define NETWORK_H

#include <cmath>

class Network
{
    friend class NetworkTrainer;
public:

    float sigmoid(float x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    struct NetworkSettings
    {
        int numHidden;
        int numInput;
        int numOutput;
    };
    Network(NetworkSettings);
    ~Network();
    void initWeights();

    void evaluate(float inputs[]);

private:

    int getIHIndex(int i, int h) { return i * numHidden + h; }
    int getHOIndex(int h, int o) { return h * numOutput + o; }

    int numHidden;
    int numInput;
    int numOutput;

    float * inputs;
    float * hiddens;
    float * outputs;
    float * clampedOutputs;

    float * weightsIH;
    float * weightsHO;
};

#endif
