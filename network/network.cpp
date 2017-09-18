#include "network.h"
#include <assert.h>
#include <stdlib.h>
#include <cstring>
#include <random>
#include <iostream>
#include <fstream>

Network::Network(Settings settings) :
    numInputs(settings.numInputs),
    numHidden(settings.numHidden),
    numOutputs(settings.numOutputs),
    func(settings.func)
{
    assert(settings.numInputs > 0 && settings.numOutputs > 0 && settings.numHidden > 0);
    initNetwork();
    initWeights();
}

void Network::initNetwork()
{
    int totalNumInputs = numInputs + 1;
    int totalNumHidden = numInputs + 1;

    inputNeurons.resize(totalNumInputs);
    hiddenNeurons.resize(totalNumHidden);
    outputNeurons.resize(numOutputs);
    clampedOutputs.resize(numOutputs);

    weightsIH.resize(totalNumInputs * totalNumHidden);
    weightsHO.resize(totalNumHidden * numOutputs);

    memset(inputNeurons.data(), 0, inputNeurons.size() * sizeof(double));
    memset(hiddenNeurons.data(), 0, hiddenNeurons.size() * sizeof(double));
    memset(outputNeurons.data(), 0, outputNeurons.size() * sizeof(double));
    memset(clampedOutputs.data(), 0, clampedOutputs.size() * sizeof(double));

    inputNeurons.back() = -1.0;
    hiddenNeurons.back() = -1.0;

    memset(weightsIH.data(), 0, weightsIH.size() * sizeof(double));
    memset(weightsHO.data(), 0, weightsHO.size() * sizeof(double));
}

void Network::initWeights()
{
    double minWeight = -0.5;
    double maxWeight = 0.5;

    std::uniform_real_distribution<double> unif(minWeight, maxWeight);
    std::default_random_engine re;

    for(int i = 0; i <= numInputs; i++)
    {
        for(int h = 0; h <= numHidden; h++)
        {
            int index = getIHIndex(i, h);
            weightsIH[index] = unif(re);
        }
    }

    for(int h = 0; h <= numHidden; h++)
    {
        for(int o = 0; o < numOutputs; o++)
        {
            int index = getHOIndex(h, o);
            weightsHO[index] = unif(re);
        }
    }
}

std::vector<double> Network::evaluate(std::vector<double> inputs)
{
    if(inputs.size() != numInputs)
    {
        std::cout << "Error: Invalid number of inputs! (" << inputs.size() << " != " << numInputs << ")" << std::endl;
        std::vector<double> tmp;
        for(int i = 0; i < numOutputs; i++)
            tmp.push_back(0);
        return tmp;
    }

    for(int i = 0; i < numInputs; i++)
    {
        inputNeurons[i] = inputs[i];
    }

    for(int h = 0; h < numHidden; h++)
    {
        double weightedSum = 0;
        for(int i = 0; i <= numInputs; i++)
        {
            int index = getIHIndex(i, h);
            weightedSum += weightsIH[index] * inputNeurons[i];
        }

        double d;
        switch(func)
        {
            case Sigmoid:
                d = sigmoidActivationFunction(weightedSum);
                break;
            case Linear:
                d = linearActivationFunction(weightedSum);
                break;
            case Step:
                d = stepActivationFunction(weightedSum);
                break;
        }

        hiddenNeurons[h] = d;
    }

    for(int o = 0; o < numOutputs; o++)
    {
        double weightedSum = 0;
        for(int h = 0; h <= numHidden; h++)
        {
            int index = getHOIndex(h, o);
            weightedSum += weightsHO[index] * hiddenNeurons[h];
        }

        double d;
        switch(func)
        {
            case Sigmoid:
                d = sigmoidActivationFunction(weightedSum);
                break;
            case Linear:
                d = linearActivationFunction(weightedSum);
                break;
            case Step:
                d = stepActivationFunction(weightedSum);
                break;
        }

        outputNeurons[o] = d;

        if(func == Sigmoid || func == Step)
        {
            if(d >= 0.5) clampedOutputs[o] = 1;
            else clampedOutputs[o] = 0;
        }
        else if(func == Linear)
        {
            clampedOutputs[o] = (int)d;
        }
    }

    return clampedOutputs;
}

void Network::saveModel(std::string filename)
{
    std::ofstream file(filename);
    if(file.is_open())
    {
        file << numInputs << " " << numHidden << " " << numOutputs << "\n";
        for(int i = 0; i <= numInputs; i++)
        {
            for(int h = 0; h <= numHidden; h++)
            {
                int index = getIHIndex(i, h);
                file << weightsIH[index] << " ";
            }
            file << "\n";
        }

        for(int h = 0; h <= numHidden; h++)
        {
            for(int o = 0; o < numOutputs; o++)
            {
                int index = getHOIndex(h, o);
                file << weightsHO[index] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Error opening file \"" << filename << "\"" << std::endl;
    }
}

Network* Network::loadModel(std::string filename)
{
    Network * net;
    std::ifstream file(filename);

    if(file.is_open())
    {
        std::string tmp;
        file >> tmp;
        net->numInputs = atoi(tmp.c_str());
        file >> tmp;
        net->numHidden = atoi(tmp.c_str());
        file >> tmp;
        net->numOutputs = atoi(tmp.c_str());

        net->initNetwork();

        for(int i = 0; i <= net->numInputs; i++)
        {
            for(int h = 0; h <= net->numHidden; h++)
            {
                file >> tmp;
                int index = net->getIHIndex(i, h);
                net->weightsIH[index] = atof(tmp.c_str());
            }
        }

        for(int h = 0; h <= net->numHidden; h++)
        {
            for(int o = 0; o < net->numOutputs; o++)
            {
                file >> tmp;
                int index = net->getHOIndex(h, o);
                net->weightsHO[index] = atof(tmp.c_str());
            }
        }
        file.close();
        return net;
    }
    else
    {
        std::cout << "Error opening file \"" << filename << "\"" << std::endl;
        return nullptr;
    }
}
