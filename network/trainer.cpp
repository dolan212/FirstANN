#include "trainer.h"
#include <assert.h>
#include <cstring>
#include <iostream>
#include <iomanip>

NetworkTrainer::NetworkTrainer(Settings sett, Network * n) :
    learningRate(sett.learningRate),
    momentum(sett.momentum),
    useBatchLearning(sett.batch),
    maxEpochs(sett.maxEpochs),
    desiredAccuracy(sett.desiredAccuracy),
    network(n),
    currentEpoch(0)
{
    assert(network != nullptr);

    std::cout << "Here?" << std::endl;

    int ih = network->weightsIH.size();
    int ho = network->weightsHO.size();
    std::cout << "Maybe Here?" << std::endl;
    std::cout << ih << std::endl;
    std::cout << ho << std::endl;
    deltaIH = new double[ih];
    std::cout << "Or Here?" << std::endl;
    deltaHO = new double[ho];

    errorGradientsHidden.resize(network->hiddenNeurons.size());
    errorGradientsOutput.resize(network->outputNeurons.size());

    memset((void*)deltaIH, 0, ih * sizeof(double));
    memset((void*)deltaHO, 0, ho * sizeof(double));

    memset(errorGradientsHidden.data(), 0, sizeof(double) * errorGradientsHidden.size());
    memset(errorGradientsOutput.data(), 0, sizeof(double) * errorGradientsOutput.size());
}

double NetworkTrainer::getHiddenErrorGradient(int hidden)
{
    double weightedSum = 0;
    for(int o = 0; o < network->numOutputs; o++)
    {
        int index = network->getHOIndex(hidden, o);
        weightedSum += network->weightsHO[index] * errorGradientsOutput[o];
    }

    return network->hiddenNeurons[hidden] * (1 - network->hiddenNeurons[hidden]) * weightedSum;
}

void NetworkTrainer::backPropagate(std::vector<double> expectedOutputs)
{
    for(int o = 0; o < network->numOutputs; o++)
    {
        errorGradientsOutput[o] = getOutputErrorGradient(expectedOutputs[o], network->outputNeurons[o]);
        for(int h = 0; h <= network->numHidden; h++)
        {
            int index = network->getHOIndex(h, o);

            if(useBatchLearning) deltaHO[index] += learningRate * network->hiddenNeurons[h] * errorGradientsOutput[o];
            else deltaHO[index] = learningRate * network->hiddenNeurons[h] * errorGradientsOutput[o] + momentum * deltaHO[index];
        }
    }

    for(int h = 0; h <= network->numHidden; h++)
    {
        errorGradientsHidden[h] = getHiddenErrorGradient(h);
        for(int i = 0; i <= network->numInputs; i++)
        {
            int index = network->getIHIndex(i, h);

            if(useBatchLearning) deltaIH[index] += learningRate * network->inputNeurons[i] * errorGradientsHidden[h];
            else deltaIH[index] = learningRate * network->inputNeurons[i] * errorGradientsHidden[h] + momentum * deltaIH[index];
        }
    }
    if(!useBatchLearning)
        updateWeights();
}

void NetworkTrainer::updateWeights()
{
    for(int i = 0; i <= network->numInputs; i++)
    {
        for(int h = 0; h <= network->numHidden; h++)
        {
            int index = network->getIHIndex(i, h);
            network->weightsIH[index] += deltaIH[index];
            if(useBatchLearning)
                deltaHO[index] = 0;
        }
    }

    for(int h = 0; h <= network->numHidden; h++)
    {
        for(int o = 0; o < network->numOutputs; o++)
        {
            int index = network->getHOIndex(h, o);
            network->weightsHO[index] += deltaHO[index];
            if(useBatchLearning)
                deltaHO[index] = 0;
        }
    }
}

double NetworkTrainer::runEpoch(TrainingSet trainingSet)
{
    double accuracy = 0;
    for(TrainingEntry entry : trainingSet)
    {
        std::vector<double> out = network->evaluate(entry.inputs);
        backPropagate(entry.expectedOut);

        //std::cout << out[0] << '\r' << std::flush;

        bool correct = true;
        for(int o = 0; o < network->numOutputs; o++)
        {
            if(out[o] != entry.expectedOut[o])
            {
                correct = false;
                break;
            }
        }

        if(correct) accuracy++;
    }

    if(useBatchLearning)
        updateWeights();

    accuracy = (accuracy / (double)trainingSet.size()) * 100;

    return accuracy;
}

void NetworkTrainer::train(TrainingData data)
{
    std::cout << std::right;
    std::cout << "==============================" << std::endl;
    std::cout << "|" << std::setw(10) << "Beginning Training" << std::setw(10) << "|" << std::endl;
    std::cout << "|" << std::setw(10) << "------------------" << std::setw(10) << "|" << std::endl;
    std::cout << "|" << std::setw(10) << "desiredAccuracy = " << desiredAccuracy << std::setw(10) << "|" << std::endl;
    std::cout << "|" << std::setw(10) << "maxEpochs = " << maxEpochs << std::setw(10) << "|" << std::endl;
    std::cout << "==============================" << std::endl;

    double trainingAccuracy = 0;
    while(currentEpoch < maxEpochs && trainingAccuracy < desiredAccuracy)
    {
        trainingAccuracy = runEpoch(data.trainingSet);
        double genAccuracy = runEpoch(data.generalizationSet);

        std::cout << "Epoch: " << currentEpoch
                  << " | Training Set Accuracy: " << trainingAccuracy
                  << " | Generalization Set Accuracy: " << genAccuracy
                  << std::endl;

        currentEpoch++;
    }

    std::cout << "===========================" << std::endl;
    std::cout << "|    Training Complete    |" << std::endl;
    std::cout << "===========================" << std::endl;

    double valAccuracy = 0;
    for(TrainingEntry entry : data.validationSet)
    {
        std::vector<double> out = network->evaluate(entry.inputs);

        bool correct = true;
        for(int o = 0; o < network->numOutputs; o++)
        {
            if(out[o] != entry.expectedOut[o])
            {
                correct = false;
                break;
            }
        }

        if(correct) valAccuracy++;
    }

    valAccuracy  = (valAccuracy / (double)data.validationSet.size()) * 100;

    std::cout << "Validation Set Accuracy: " << valAccuracy << std::endl;
}
