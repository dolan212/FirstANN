#ifndef TRAINING_DATA_READER_H
#define TRAINING_DATA_READER_H

#include "../network/trainer.h"
#include <string>

class TrainingDataReader
{
public:

    TrainingDataReader(std::string const& filename);
    bool readData();

    inline int getNumInputs() const { return numInputs; }
    inline int getNumOutputs() const { return numOutputs; }

    inline int getNumTrainingSets() const { return 0; }

    TrainingData const& getTrainingData() const { return data; }
private:
    void createTrainingData();

    std::string filename;
    int numInputs;
    int numOutputs;

    std::vector<TrainingEntry> entries;
    TrainingData data;
};

#endif
