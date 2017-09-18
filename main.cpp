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

    Network::Settings netSettings;
    netSettings.numInputs = numIn;
    netSettings.numHidden = numHidden;
    netSettings.numOutputs = numOut;
    netSettings.func = Sigmoid;

    Network * net = new Network(netSettings);

    NetworkTrainer::Settings trainerSettings;
    trainerSettings.learningRate = 0.9;
    trainerSettings.maxEpochs = 2000;
    NetworkTrainer trainer(trainerSettings, net);

    trainer.train(reader.getTrainingData());

    cout << "Enter filename to save model: ";
    cin >> filename;

    net->saveModel(filename);

}
