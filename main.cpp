#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h> 
#include <queue>

using namespace std;

struct Node
{
    double collector;
    vector<Node *> connections;
    vector<double> weights;
    double error;

    Node(vector<Node *> connections)
    {
        this->connections = connections;
        this->collector = 0;
        for (int i = 0; i < connections.size(); i++)
        {
            weights.push_back((double)rand() / RAND_MAX);
        }
    }

    Node()
    {
        this->collector = 0;
    }
};

double waveFunction(double x) 
{
    return (cos(x) + 1) * 0.5;
}

class NeuralNetwork
{
private:
    vector<int> structure;
    vector<vector<Node *>> network;
    vector<vector<double>> inputs;

    double sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoid_derivative(double collector)
    {
        return collector * (1.0 - collector);
    }

    // update the weights
    void update_weights(double l_rate)
    {
        for (int i = 1; i < network.size(); i++)
        {
            vector<double> inputs;
            for (auto neuron : network[i - 1])
            {
                inputs.push_back(neuron->collector);
            }
            for (auto neuron : network[i])
            {
                for (int j = 0; j < inputs.size(); j++)
                {
                    neuron->weights[j] -= l_rate * neuron->error * inputs[j];
                }
                // neuron->weights.back() -= l_rate * neuron->error;
            }
        }
    }

    // feed forward the inputs
    void feed_forward(vector<double> row)
    {
        for (int i = 0; i < structure[0]; i++)
        {
            network[0][i]->collector = row[i];
        }
        for (int i = 1; i < structure.size(); i++)
        {
            for (int j = 0; j < structure[i]; j++)
            {
                double sum = 0;
                for (int k = 0; k < structure[i - 1]; k++)
                {
                    sum += network[i - 1][k]->collector * network[i][j]->weights[k];
                }
                network[i][j]->collector = sigmoid(sum);
            }
        }
    }

    // back propagate the error
    void back_propagate(vector<double> expected)
    {
        for (int i = network.size() - 1; i > 0; i--)
        {
            vector<Node *> layer = network[i];
            vector<double> errors;

            if (i == network.size() - 1)
            {
                for (int j = 0; j < layer.size(); j++)
                {
                    errors.push_back(layer[j]->collector - expected[j]);
                }
            }
            else
            {
                for (int j = 0; j < layer.size(); j++)
                {
                    double error = 0.0;
                    for (auto node : network[i + 1])
                    {
                        error += node->weights[j] * node->error;
                    }
                    errors.push_back(error);
                }
            }
            // 
            for (int j = 0; j < layer.size(); j++)
            {
                layer[j]->error = errors[j] * sigmoid_derivative(layer[j]->collector);
            }

            errors.clear(); // Clear the vector for the next iteration
        }
    }
public:
    NeuralNetwork(string structureFile)
    {
        // Load structure from file into the structure vector
        ifstream file(structureFile);
        if (file.is_open())
        {
            char c;
            string val;
            while (file >> c)
            {
                if (c == ',')
                {
                    structure.push_back(stoi(val));
                    val = "";
                }
                else
                {
                    val += c;
                }
            }
            structure.push_back(stoi(val));
        }
        file.close();

        // Create network with connections to each node based on structure
        for (int i = 0; i < structure.size(); i++)
        {
            vector<Node *> layer;
            for (int j = 0; j < structure[i]; j++)
            {
                if (i == 0)
                {
                    layer.push_back(new Node());
                }
                else
                {
                    layer.push_back(new Node(network[i - 1]));
                }
            }
            network.push_back(layer);
        }
    }

    // train the neural network
    void train(vector<vector<double>> inputs, int num_epochs, double target_error = 0.05, double l_rate = 0.1)
    {
        int num_inputs = network[0].size();

        cout << "Number of inputs = " << inputs.size() << endl;
        // train the network
        for (int epoch = 0; epoch < num_epochs; epoch++)
        {
            double epoch_error = 0;
            for (auto row : inputs)
            {
                vector<double> expected;
                feed_forward(row);
                for (int j = 0; j < structure[structure.size() - 1]; j++)
                {
                    expected.push_back(row[num_inputs + j]);
                }
                for (int j = 0; j < structure[structure.size() - 1]; j++)
                {
                    epoch_error += pow(network[structure.size() - 1][j]->collector - expected[j], 2);
                }
                back_propagate(expected);
                update_weights(l_rate);
            }
            if (epoch_error <= target_error)
            {
                cout << "Target error reached error " << epoch_error << endl;
                return;
            }
            cout << setprecision(10) << ">Epoch=" << epoch << " l_rate=" << l_rate << " error=" << setprecision(20) << epoch_error << endl;
        }
    }

    // run with a vector of inputs and print output
    void run(vector<double> inputs)
    {
        feed_forward(inputs);
        for (int i = 0; i < structure[structure.size() - 1]; i++)
        {
            cout << network[structure.size() - 1][i]->collector << " ";
        }
        cout << endl;
    }

    void run_generative(vector<double> inputs, int num_generations)
    {
        deque<double> input_queue;
        double output;
        cout << fixed;
        // initialize input queue
        for(int i = 0; i < structure[0]; i++)
        {
            input_queue.push_back(inputs[i]);
        }
        double x = 0;
        for(int i = 0; i < num_generations; i++)
        {
            // load input queue into nn
            feed_forward({input_queue.begin(), input_queue.end()});
            output = get_output();
            cout << setprecision(1) << "f(" << x << "): " << setprecision(4) << output << endl;
            input_queue.pop_front();
            input_queue.push_back(output);
            x += 0.1;
        }
    }

    double get_output()
    {
        return network[structure.size() - 1][0]->collector;
    }

    // save the weights to a file
    void save(string filename)
    {
        ofstream file;
        file.open(filename);
        for (int i = 1; i < structure.size(); i++)
        {
            for (int j = 0; j < structure[i]; j++)
            {
                for (int k = 0; k < structure[i - 1]; k++)
                {
                    file << network[i][j]->weights[k] << " ";
                }
                file << endl;
            }
        }
        file.close();
    }
};



// main function
int main()
{
    // seed random number generator
    srand(time(NULL));
    
    // create the neural network
    NeuralNetwork nn("network.csv");

    // generate inputs for the neural network incrementing by 0.1
    vector<vector<double>> inputs;
    for(int i = -400; i < 400; i++)
    {
        vector<double> input;
        for(double j = 0; j <= 1; j += 0.1)
        {
            input.push_back(waveFunction(i+j));
        }
        inputs.push_back(input);
    }

    // train the neural network
    nn.train(inputs, 100000, 0.01, 0.2);

    // test the neural network with the inputs -1 through 0
    vector<double> input;
    for (double i = -1; i <= 0; i += 0.1)
    {
        input.push_back(waveFunction(i));
    }
    nn.run_generative(input, 200);
    

    return 0;
}
