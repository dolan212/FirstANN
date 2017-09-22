#ifndef TRAINER_H
#define TRAINER_H

#include "network.h"
#include <vector>

struct TrainingEntry
{
    std::vector<float> inputs;
    std::vector<float> expectedOut;
};

typedef std::vector<TrainingEntry> TrainingSet;

struct TrainingData
{
    TrainingSet trainingSet;
    TrainingSet generalizationSet;
    TrainingSet validationSet;
};

class NetworkTrainer
{
public:
    struct TrainerSettings
    {
        double learningRate = 0.01;
        double momentum = 0.9;

        int maxEpochs = 150;
        int desiredAccuracy = 90;
    };

    NetworkTrainer(TrainerSettings, Network *);
    ~NetworkTrainer();

    void train(TrainingData);
private:
    inline float getOutputErrorGradient(float desiredValue, float outputValue)
    {
        return outputValue * (1.0 - outputValue) * (desiredValue - outputValue);
    }

    float getHiddenErrorGradient(int hidden);

    float runEpoch(TrainingSet);

    void backPropagate(std::vector<float>);
    void updateWeights();

    Network* network;

    float learningRate;
    float momentum;
    int desiredAccuracy;
    int maxEpochs;

    float * deltaIH;
    float * deltaHO;
    float * errorGradientsHidden;
    float * errorGradientsOutput;

    int currentEpoch;
};

#endif
