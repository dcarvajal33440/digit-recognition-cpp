#include <iostream>
#include <vector>
#include <cmath>

double sigmoid(double wsum){
	return 1.0/ (1.0 + std::exp(-wsum));
}

double predict(const std::vector<double>& inputs, const std::vector<double>& weights
		, double bias){
	double wsum = bias;
	for(std::size_t i = 0; i< inputs.size(); ++i){
		wsum += inputs[i] * weights[i];
	}
	return sigmoid(wsum);
}
int main() {
	// Inputs and the parameters our neuron will learn.
	std::vector<std::vector<double>> training_inputs = {
	{0.0, 0.0}, {0.0,1.0},{1.0,0.0},{1.0,1.0}
	};
    	std::vector<double> weights = {0.2, 0.4};
    	double bias = 0.5;

    	// Training settings.
	std::vector<double> targets = {0.0,0.0,0.0,1.0};
    	double learning_rate = 0.1;
    	double loss_threshold = 0.0001;
    	int max_iterations = 10000;

    	for (int epoch = 0; epoch < max_iterations; ++epoch) {
   		 // Train on each example using the same weights and bias.
    		for (std::size_t sample = 0; sample < training_inputs.size(); ++sample) {
        		const std::vector<double>& inputs = training_inputs[sample];
        		double target = targets[sample];

        		double activation = predict(inputs, weights, bias);
        		double error = activation - target;
        		double sigmoid_derivative = activation * (1.0 - activation);
        		double loss_gradient = error * sigmoid_derivative;

        		for (std::size_t i = 0; i < weights.size(); ++i) {
            			double weight_gradient = loss_gradient * inputs[i];
            			weights[i] -= learning_rate * weight_gradient;
        		}

        		bias -= learning_rate * loss_gradient;
    		}

    		// Evaluate all examples without changing the parameters.
    		double total_loss = 0.0;

    		for (std::size_t sample = 0; sample < training_inputs.size(); ++sample) {
        		double activation = predict(training_inputs[sample], weights, bias);
        		double error = activation - targets[sample];
        		total_loss += 0.5 * error * error;
    		}

    		double average_loss = total_loss / training_inputs.size();

    		if (epoch % 100 == 0 || average_loss < loss_threshold || epoch == max_iterations - 1) {
        		std::cout << "Epoch: " << epoch + 1
                  	<< " | Average loss: " << average_loss << '\n';
    		}

    		if (average_loss < loss_threshold) {
        		std::cout << "Loss target reached.\n";
        		break;
    		}

    		if (epoch == max_iterations - 1) {
        		std::cout << "Epoch limit reached.\n";
    		}
	}
	int correct = 0;

	std::cout << "\nFinal predictions:\n";

	for (std::size_t sample = 0; sample < training_inputs.size(); ++sample) {
    		double output = predict(training_inputs[sample], weights, bias);

    		int predicted = 0;
    		if (output >= 0.5) {
        		predicted = 1;
    		}

    		if (predicted == targets[sample]) {
        		++correct;
    		}

    		std::cout << training_inputs[sample][0] << " AND "
              	<< training_inputs[sample][1]
              	<< " | Output: " << output
              	<< " | Predicted: " << predicted
              	<< " | Target: " << targets[sample] << '\n';
	}

	double accuracy = 100.0 * correct / training_inputs.size();
	std::cout << "Training accuracy: " << accuracy << "%\n";
	return 0;
}
