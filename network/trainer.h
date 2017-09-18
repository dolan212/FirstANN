#ifndef TRAINER_H
#define TRAINER_H

#include "network.h"
#include <vector>
#include <iostream>


struct TrainingEntry
{
    std::vector<double> inputs;
    std::vector<double> expectedOut;
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
    struct Settings
    {
        double learningRate = 0.01;
        double momentum = 0.9;
        bool batch = false;

        int maxEpochs = 150;
        int desiredAccuracy = 90;
    };

    NetworkTrainer(Settings, Network *);//

    void train(TrainingData);

private:
    inline double getOutputErrorGradient(double desiredValue, double outputValue) const//
    {
        return outputValue * (1.0 - outputValue) * (desiredValue - outputValue);
    }

    double getHiddenErrorGradient(int hidden);//

    double runEpoch(TrainingSet);//

    void backPropagate(std::vector<double>);//
    void updateWeights();//

    Network* network;

    double learningRate;
    double momentum;
    int desiredAccuracy;
    int maxEpochs;
    bool useBatchLearning;

    double * deltaIH;
    double * deltaHO;
    std::vector<double> errorGradientsHidden;
    std::vector<double> errorGradientsOutput;

    int currentEpoch;
};

#endif
