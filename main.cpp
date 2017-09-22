#include <iostream>
#include "reader/reader.h"
#include "network/trainer.h"
#include "network/network.h"

using namespace std;
int main()
{
    int numHidden;
    string filename;
    cout << "Enter filename: ";
    cin >> filename;
    cout << "Number of hiddens: ";
    cin >> numHidden;

    TrainingDataReader reader(filename);
    if(!reader.readData()) return 0;
    int numIn = reader.getNumInputs();
    int numOut = reader.getNumOutputs();

    Network::NetworkSettings netSettings;
    netSettings.numInput = numIn;
    netSettings.numHidden = numHidden;
    netSettings.numOutput = numOut;

    Network * net = new Network(netSettings);

    NetworkTrainer::TrainerSettings trainerSettings;
    trainerSettings.learningRate = 0.9;
    trainerSettings.maxEpochs = 200;
    NetworkTrainer trainer(trainerSettings, net);

    trainer.train(reader.getTrainingData());
    delete net;
}
