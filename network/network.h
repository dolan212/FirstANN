#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include <cmath>

enum ActivationFunction {
    Sigmoid, Linear, Step
};

class Network
{
    friend class NetworkTrainer;

public:
    inline static double sigmoidActivationFunction(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    inline static double stepActivationFunction(double x) { return (x <= 0) ? 0 : 1; }
    inline static double linearActivationFunction(double x) { return x; }

    struct Settings {
        int numInputs;
        int numHidden = 15;
        int numOutputs;
        ActivationFunction func = Sigmoid;
    };

    Network(Settings);
    Network();
    std::vector<double> evaluate(std::vector<double>);
    void saveModel(std::string);
    static Network* loadModel(std::string);

private:
    void initNetwork();
    void initWeights();

    int getIHIndex(int i, int h) { return i * numHidden + h; }
    int getHOIndex(int h, int o) { return h * numOutputs + o; }

    int numInputs;
    int numHidden;
    int numOutputs;

    std::vector<double> inputNeurons;
    std::vector<double> hiddenNeurons;
    std::vector<double> outputNeurons;

    std::vector<double> clampedOutputs;

    std::vector<double> weightsIH;
    std::vector<double> weightsHO;

    ActivationFunction func;
};

#endif
