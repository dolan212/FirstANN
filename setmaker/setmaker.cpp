#include <iostream>
#include "../split.h"
#include <fstream>

using namespace std;

int main(int argc, char* argv[])
{
    string filename;
    if(argc > 1) filename = argv[1];
    else
    {
        cout << "Enter filename: ";
        cin >> filename;
    }

    ifstream inFile(filename);
    if(inFile.is_open())
    {
        cout << "Opened file" << endl;
        cout << "Reading data..." << endl;
        string line;
        inFile >> line;

        vector<string> columns = line | ',';

        vector<vector<string>> data;

        /*while(inFile >> line)
            data.push_back(line | ',');

        inFile.close();*/


        cout << "The available columns are:" << endl;
        cout << "| ";
        for(int i = 0; i < columns.size(); i++)
            cout << columns[i] << " (" << i << ") | ";
        cout << endl;

        cout << "Enter columns to remove (comma seperated): ";
        string removeS;
        cin >> removeS;

        vector<string> r = removeS | ',';
        vector<int> remove;
        if(removeS != "-1")
        {
            for(string s : r)
            {
                if(s.find("-") != string::npos)
                {
                    vector<string> tmp = s | '-';
                    int start = atoi(tmp[0].c_str());
                    int end = atoi(tmp[1].c_str());
                    for(int i = start; i <= end; i++)
                        remove.push_back(i);
                }
                else
                {
                    remove.push_back(atoi(s.c_str()));
                }
            }
        }

        cout << "Removing columns" << endl;
        for(int i = 0; i < columns.size(); i++)
        {
            for(int j = 0; j < remove.size(); j++)
            {
                if(i == remove[j])
                {
                    columns[i] = "";
                    break;
                }
            }
        }

        cout << "Transferring data" << endl;
        while(inFile >> line)
        {
            vector<string> tmp = line | ',';
            vector<string> tmp2;
            for(int i = 0; i < tmp.size(); i++)
            {
                bool toRemove = false;
                for(int j : remove)
                {
                    if(i == j)
                    {
                        toRemove = true;
                        break;
                    }
                }
                if(!toRemove) tmp2.push_back(tmp[i]);
            }
            data.push_back(tmp2);
        }


        vector<string> newcols;
        for(string s : columns)
        {
            if(s != "") newcols.push_back(s);
        }


        cout << "New columns are: " << endl << "| ";
        for(int i = 0; i < newcols.size(); i++)
        {
            cout << newcols[i] << " (" << i << ") | ";
        }
        cout << endl;

        vector<int> inputs;
        cout << "Enter input columns: ";
        cin >> line;
        for(string s : line | ',')
        {
            if(s.find("-") != string::npos)
            {
                vector<string> tmp = s | '-';
                int start = atoi(tmp[0].c_str());
                int end = atoi(tmp[1].c_str());
                for(int i = start; i <= end; i++)
                    inputs.push_back(i);
            }
            else
            {
                inputs.push_back(atoi(s.c_str()));
            }
        }

        vector<int> outputs;
        cout << "Enter output columns: ";
        cin >> line;
        for(string s : line | ',')
        {
            if(s.find("-") != string::npos)
            {
                vector<string> tmp = s | '-';
                int start = atoi(tmp[0].c_str());
                int end = atoi(tmp[1].c_str());
                for(int i = start; i <= end; i++)
                    outputs.push_back(i);
            }
            else
            {
                outputs.push_back(atoi(s.c_str()));
            }
        }

        struct Range {
            double start;
            double end;
        };

        cout << "Calculating ranges" << endl;
        vector<string> inRanges;
        for(int i = 0; i < inputs.size(); i++)
        {
            double max = atof(data[0][inputs[i]].c_str());
            double min = max;
            for(int j = 0; j < data.size(); j++)
            {
                double weh = atof(data[j][inputs[i]].c_str());
                if(weh > max) max = weh;
                if(weh < min) min = weh;
            }
            string s = to_string(min) + "-" + to_string(max);
            inRanges.push_back(s);
        }

        cout << "Enter output file name: ";
        cin >> filename;

        ofstream outFile(filename);

        if(outFile.is_open())
        {
            line = newcols[0];
            for(int i = 1; i < newcols.size(); i++)
                line += "," + newcols[i];
            line += "\n";
            outFile << line;

            line = to_string(inputs[0]);
            for(int i = 1; i < inputs.size(); i++)
                line += "," + to_string(inputs[i]);
            line += "\n";
            outFile << line;

            line = to_string(outputs[0]);
            for(int i = 1; i < outputs.size(); i++)
                line += "," + to_string(outputs[i]);
            line += "\n";
            outFile << line;

            line = inRanges[0];
            for(int i = 1; i < inRanges.size(); i++)
                line += "," + inRanges[i];
            line += "\n";
            outFile << line;

            for(int i = 0; i < data.size(); i++)
            {
                line = data[i][0];
                for(int j = 1; j < data[i].size(); j++)
                {
                    line += "," + data[i][j];
                }
                line += "\n";
                outFile << line;
            }


            outFile.close();

            cout << "Data written successfully to \"" << filename << "\"" << endl;
        }
        else
        {
            cout << "Error opening file\"" << filename << "\"" << endl;
            return 1;
        }

    }
    else
    {
        cout << "Error opening file \"" << filename << "\"" << endl;
        return 1;
    }

    return 0;
}
