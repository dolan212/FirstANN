#include "network.h"
#include <random>

Network::Network(NetworkSettings sett)
{
    numInput = sett.numInput;
    numOutput = sett.numOutput;
    numHidden = sett.numHidden;

    int totalIn = numInput + 1;
    int totalHid = numHidden + 1;

    inputs = new float[totalIn];
    hiddens = new float[totalHid];
    outputs = new float[numOutput];
    clampedOutputs = new float[numOutput];

    weightsIH = new float[totalIn * totalHid];
    weightsHO = new float[totalHid * numOutput];

    for(int i = 0; i < totalIn; i++)
        inputs[i] = 0;

    for(int h = 0; h < totalHid; h++)
        hiddens[h] = 0;

    for(int o = 0; o < numOutput; o++)
    {
        outputs[o] = 0;
        clampedOutputs[o] = 0;
    }

    for(int i = 0; i < totalIn * totalHid; i++)
        weightsIH[i] = 0;

    for(int i = 0; i < numOutput * totalHid; i++)
        weightsHO[i] = 0;

    initWeights();
}

void Network::initWeights()
{
    float minWeight = -0.5;
    float maxWeight = 0.5;

    std::uniform_real_distribution<float> unif(minWeight, maxWeight);
    std::default_random_engine re;

    for(int i = 0; i <= numInput; i++)
    {
        for(int h = 0; h <= numHidden; h++)
        {
            int index = getIHIndex(i, h);
            weightsIH[index] = unif(re);
        }
    }

    for(int h = 0; h <= numHidden; h++)
    {
        for(int o = 0; o < numOutput; o++)
        {
            int index = getHOIndex(h, o);
            weightsHO[index] = unif(re);
        }
    }
}

void Network::evaluate(float input[])
{
    for(int i = 0; i < numInput; i++)
    {
        inputs[i] = input[i];
    }

    for(int h = 0; h < numHidden; h++)
    {
        float weightedSum = 0;
        for(int i = 0; i <= numInput; i++)
        {
            int index = getIHIndex(i, h);
            weightedSum += weightsIH[index] * inputs[i];
        }
        hiddens[h] = sigmoid(weightedSum);
    }

    for(int o = 0; o < numOutput; o++)
    {
        float weightedSum = 0;
        for(int h = 0; h <= numHidden; h++)
        {
            int index = getHOIndex(h, o);
            weightedSum += weightsHO[index] * hiddens[h];
        }
        outputs[o] = sigmoid(weightedSum);
        if(outputs[o] >= 0.5) clampedOutputs[o] = 1;
        else clampedOutputs[o] = 0;
    }
}


Network::~Network()
{
    delete [] inputs;
    delete [] outputs;
    delete [] hiddens;

    delete [] weightsIH;
    delete [] weightsHO;
}
