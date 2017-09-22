#include "reader.h"
#include "../split.h"
#include <assert.h>
#include <algorithm>
#include <string.h>
#include <fstream>
#include <iostream>

TrainingDataReader::TrainingDataReader(std::string const& filename) :
    filename(filename)
{
    assert(!filename.empty());
}

bool TrainingDataReader::readData()
{

    struct Range {
        float start;
        float end;
    };
    assert(!filename.empty());
    std::cout << "Reading data..." << std::endl;

    std::ifstream inputFile;
    inputFile.open(filename);

    if(inputFile.is_open())
    {
        std::string line;
        inputFile >> line;

        inputFile >> line;
        std::vector<int> inputIndices;
        for(std::string s : line | ',')
            inputIndices.push_back(atoi(s.c_str()));

        inputFile >> line;
        std::vector<int> outputIndices;
        for(std::string s : line | ',')
            outputIndices.push_back(atoi(s.c_str()));

        numInputs = inputIndices.size();
        numOutputs = outputIndices.size();

        inputFile >> line;
        std::vector<Range> inRanges;
        for(std::string s : line | ',')
        {
            if(s.find("-") == std::string::npos)
            {
                std::cout << "Error: invalid range \"" << s << "\"" << std::endl;
                return false;
            }
            std::vector<std::string> startAndEnd = s | '-';
            Range range;
            range.start = atof(startAndEnd[0].c_str());
            range.end = atof(startAndEnd[1].c_str());

            inRanges.push_back(range);
        }


        while(inputFile >> line)
        {
            std::vector<std::string> data = line | ',';

            std::vector<float> outputs;
            std::vector<float> inputs;
            for(int i : outputIndices)
                outputs.push_back(atof(data[i].c_str()));

            int rangeIn = 0;
            for(int i : inputIndices)
            {
                float val = atof(data[i].c_str());
                Range r = inRanges[rangeIn];
                val = (val - r.start) / (r.end - r.start);
                inputs.push_back(val);
                rangeIn++;
            }

            TrainingEntry entry;
            entry.inputs = inputs;
            entry.expectedOut = outputs;
            entries.push_back(entry);
        }
        inputFile.close();

        assert(entries.size() > 0);
        createTrainingData();
        return true;
    }
    else
    {
        std::cout << "Error reading file \"" << filename << "\"" << std::endl;
        return false;
    }
}

void TrainingDataReader::createTrainingData()
{
	assert(!entries.empty());
    std::cout << "Creating training data..." << std::endl;
	std::random_shuffle(entries.begin(), entries.end());

	int numEntries = entries.size();
	int numTrainingEntries = (int) (0.6 * numEntries);
	int numGeneralizationEntries = (int) ceil(0.2 * numEntries);

	int entry = 0;

	for(; entry < numTrainingEntries; entry++)
		data.trainingSet.push_back(entries[entry]);

	for(; entry < numTrainingEntries + numGeneralizationEntries; entry++)
		data.generalizationSet.push_back(entries[entry]);

	for(; entry < numEntries; entry++)
		data.validationSet.push_back(entries[entry]);
}
