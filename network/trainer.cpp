#include "trainer.h"
#include <iostream>
#include <iomanip>

NetworkTrainer::NetworkTrainer(TrainerSettings set, Network * net)
{
    network = net;
    learningRate = set.learningRate;
    momentum = set.momentum;

    maxEpochs = set.maxEpochs;
    desiredAccuracy = set.desiredAccuracy;

    deltaIH = new float[(net->numInput + 1) * (net->numHidden + 1)];
    deltaHO = new float[(net->numHidden + 1) * net->numOutput];

    errorGradientsHidden = new float[net->numHidden + 1];
    errorGradientsOutput = new float[net->numOutput];
    currentEpoch = 0;
}

NetworkTrainer::~NetworkTrainer()
{
    delete [] deltaIH;
    delete [] deltaHO;
    delete [] errorGradientsHidden;
    delete [] errorGradientsOutput;
}

float NetworkTrainer::getHiddenErrorGradient(int hidden)
{
    float weightedSum = 0;
    for(int o = 0; o < network->numOutput; o++)
    {
        int index = network->getHOIndex(hidden, o);
        weightedSum += network->weightsHO[index] * errorGradientsOutput[o];
    }
    return network->hiddens[hidden] * (1 - network->hiddens[hidden]) * weightedSum;
}

void NetworkTrainer::backPropagate(std::vector<float> expected)
{
    for(int o = 0; o < network->numOutput; o++)
    {
        errorGradientsOutput[o] = getOutputErrorGradient(expected[o], network->outputs[o]);
        for(int h = 0; h <= network->numHidden; h++)
        {
            int index = network->getHOIndex(h, o);

            deltaHO[index] = learningRate * network->hiddens[h] * errorGradientsOutput[o] + momentum * deltaHO[index];
        }
    }

    for(int h = 0; h <= network->numHidden; h++)
    {
        errorGradientsHidden[h] = getHiddenErrorGradient(h);
        for(int i = 0; i <= network->numInput; i++)
        {
            int index = network->getIHIndex(i, h);

            deltaIH[index] = learningRate * network->inputs[h] * errorGradientsHidden[h] + momentum * deltaIH[index];
        }
    }

    updateWeights();
}

void NetworkTrainer::updateWeights()
{
    for(int i = 0; i <= network->numInput; i++)
    {
        for(int h = 0; h <= network->numHidden; h++)
        {
            int index = network->getIHIndex(i, h);
            network->weightsIH[index] += deltaIH[index];
        }
    }

    for(int h = 0; h <= network->numHidden; h++)
    {
        for(int o = 0; o < network->numOutput; o++)
        {
            int index = network->getHOIndex(h, o);
            network->weightsHO[index] += deltaHO[index];
        }
    }
}

float NetworkTrainer::runEpoch(TrainingSet trainingSet)
{
    float accuracy = 0;
    for(TrainingEntry entry : trainingSet)
    {
        float inputs[entry.inputs.size()];
        for(int i = 0; i < entry.inputs.size(); i++)
            inputs[i] = entry.inputs[i];
        network->evaluate(inputs);
        backPropagate(entry.expectedOut);

        bool correct = true;
        for(int o = 0; o < network->numOutput; o++)
        {
            if(network->clampedOutputs[o] != entry.expectedOut[o])
            {
                correct = false;
                break;
            }
        }

        if(correct)
            accuracy++;
    }

    accuracy = (accuracy / (float)trainingSet.size()) * 100;

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

    float trainingAccuracy = 0;
    while(currentEpoch < maxEpochs && trainingAccuracy < desiredAccuracy)
    {
        trainingAccuracy = runEpoch(data.trainingSet);
        float genAccuracy = runEpoch(data.generalizationSet);

        std::cout << "Epoch: " << currentEpoch
                  << " | Training Set Accuracy: " << trainingAccuracy
                  << " | Generalization Set Accuracy: " << genAccuracy
                  << std::endl;

        currentEpoch++;
    }

    std::cout << "===========================" << std::endl;
    std::cout << "|    Training Complete    |" << std::endl;
    std::cout << "===========================" << std::endl;

    float valAccuracy = 0;
    for(TrainingEntry entry : data.validationSet)
    {
        float inputs[entry.inputs.size()];
        for(int i = 0; i < entry.inputs.size(); i++)
            inputs[i] = entry.inputs[i];
        network->evaluate(inputs);

        bool correct = true;
        for(int o = 0; o < network->numOutput; o++)
        {
            if(network->clampedOutputs[o] != entry.expectedOut[o])
            {
                correct = false;
                break;
            }
        }

        if(correct) valAccuracy++;
    }

    valAccuracy  = (valAccuracy / (float)data.validationSet.size()) * 100;

    std::cout << "Validation Set Accuracy: " << valAccuracy << std::endl;
}
