#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>

class NeuralNetwork {
private:
    std::size_t input_count;
    std::vector<std::vector<double>> hidden_weights;
    std::vector<double> hidden_biases;
    std::vector<double> output_weights;
    double output_bias;

    struct Gradients {
        std::vector<std::vector<double>> hidden_weights;
        std::vector<double> hidden_biases;
        std::vector<double> output_weights;
        double output_bias;
    };

    static double sigmoid(double wsum) {
        if (wsum >= 0.0) {
            return 1.0 / (1.0 + std::exp(-wsum));
        }

        double exponential = std::exp(wsum);
        return exponential / (1.0 + exponential);
    }

    static double neuron_output(
        const std::vector<double>& inputs,
        const std::vector<double>& weights,
        double bias
    ) {
        double wsum = bias;

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            wsum += inputs[i] * weights[i];
        }

        return sigmoid(wsum);
    }

    void validate_inputs(const std::vector<double>& inputs) const {
        if (inputs.size() != input_count) {
            throw std::invalid_argument("Incorrect number of inputs.");
        }
    }

    std::vector<double> hidden_activations(
        const std::vector<double>& inputs
    ) const {
        validate_inputs(inputs);
        std::vector<double> outputs(hidden_weights.size(), 0.0);

        for (std::size_t neuron = 0; neuron < hidden_weights.size(); ++neuron) {
            outputs[neuron] = neuron_output(
                inputs, hidden_weights[neuron], hidden_biases[neuron]
            );
        }

        return outputs;
    }

    // Compute every derivative before updating any parameters.
    Gradients calculate_gradients(
        const std::vector<double>& inputs,
        double target
    ) const {
        std::vector<double> hidden_outputs = hidden_activations(inputs);
        double activation = neuron_output(
            hidden_outputs, output_weights, output_bias
        );
        double output_gradient =
            (activation - target) * activation * (1.0 - activation);

        Gradients gradients;
        gradients.hidden_weights = std::vector<std::vector<double>>(
            hidden_weights.size(), std::vector<double>(input_count, 0.0)
        );
        gradients.hidden_biases =
            std::vector<double>(hidden_biases.size(), 0.0);
        gradients.output_weights =
            std::vector<double>(output_weights.size(), 0.0);
        gradients.output_bias = output_gradient;

        for (std::size_t neuron = 0; neuron < hidden_weights.size(); ++neuron) {
            double h = hidden_outputs[neuron];
            gradients.output_weights[neuron] = output_gradient * h;

            double hidden_gradient =
                output_gradient * output_weights[neuron] * h * (1.0 - h);

            gradients.hidden_biases[neuron] = hidden_gradient;

            for (std::size_t i = 0; i < input_count; ++i) {
                gradients.hidden_weights[neuron][i] =
                    hidden_gradient * inputs[i];
            }
        }

        return gradients;
    }

    double numerical_gradient(
        double& parameter,
        const std::vector<double>& inputs,
        double target
    ) {
        double epsilon = 0.00001;
        double original_value = parameter;

        parameter = original_value + epsilon;
        double loss_plus = loss(inputs, target);

        parameter = original_value - epsilon;
        double loss_minus = loss(inputs, target);

        parameter = original_value;

        return (loss_plus - loss_minus) / (2.0 * epsilon);
    }

public:
    NeuralNetwork(
        std::size_t inputs,
        std::size_t hidden_neurons,
        unsigned int seed
    )
        : input_count(inputs),
          hidden_weights(hidden_neurons, std::vector<double>(inputs, 0.0)),
          hidden_biases(hidden_neurons, 0.0),
          output_weights(hidden_neurons, 0.0),
          output_bias(0.0) {
        if (inputs == 0 || hidden_neurons == 0) {
            throw std::invalid_argument("Layer sizes must be positive.");
        }

        std::mt19937 generator(seed);

        // Xavier uniform initialization.
        double hidden_limit = std::sqrt(
            6.0 / (static_cast<double>(inputs) + hidden_neurons)
        );
        double output_limit = std::sqrt(
            6.0 / (static_cast<double>(hidden_neurons) + 1.0)
        );

        std::uniform_real_distribution<double> hidden_distribution(
            -hidden_limit, hidden_limit
        );
        std::uniform_real_distribution<double> output_distribution(
            -output_limit, output_limit
        );

        for (std::size_t neuron = 0; neuron < hidden_neurons; ++neuron) {
            for (std::size_t i = 0; i < inputs; ++i) {
                hidden_weights[neuron][i] = hidden_distribution(generator);
            }

            output_weights[neuron] = output_distribution(generator);
        }
    }

    std::size_t parameter_count() const {
        return hidden_weights.size() * input_count
             + hidden_biases.size()
             + output_weights.size()
             + 1;
    }

    double predict(const std::vector<double>& inputs) const {
        std::vector<double> hidden_outputs = hidden_activations(inputs);
        return neuron_output(hidden_outputs, output_weights, output_bias);
    }

    double loss(const std::vector<double>& inputs, double target) const {
        double error = predict(inputs) - target;
        return 0.5 * error * error;
    }

    void train_one(
        const std::vector<double>& inputs,
        double target,
        double learning_rate
    ) {
        Gradients gradients = calculate_gradients(inputs, target);

        for (std::size_t neuron = 0; neuron < output_weights.size(); ++neuron) {
            output_weights[neuron] -=
                learning_rate * gradients.output_weights[neuron];
        }
        output_bias -= learning_rate * gradients.output_bias;

        for (std::size_t neuron = 0; neuron < hidden_weights.size(); ++neuron) {
            for (std::size_t i = 0; i < input_count; ++i) {
                hidden_weights[neuron][i] -=
                    learning_rate * gradients.hidden_weights[neuron][i];
            }

            hidden_biases[neuron] -=
                learning_rate * gradients.hidden_biases[neuron];
        }
    }

    bool check_gradients(
        const std::vector<std::vector<double>>& training_inputs,
        const std::vector<double>& targets
    ) {
        if (training_inputs.empty()
            || training_inputs.size() != targets.size()) {
            throw std::invalid_argument(
                "Provide matching, nonempty inputs and targets."
            );
        }

        for (const std::vector<double>& inputs : training_inputs) {
            validate_inputs(inputs);
        }

        int checks = 0;
        int failures = 0;
        double tolerance = 0.0000001;
        double largest_difference = 0.0;

        for (std::size_t sample = 0; sample < training_inputs.size(); ++sample) {
            const std::vector<double>& inputs = training_inputs[sample];
            double target = targets[sample];
            Gradients gradients = calculate_gradients(inputs, target);

            auto check = [&](double& parameter, double analytical) {
                double numerical = numerical_gradient(
                    parameter, inputs, target
                );
                double difference = std::abs(analytical - numerical);
                ++checks;

                if (!std::isfinite(difference) || difference > tolerance) {
                    ++failures;
                    std::cout << "Failed check " << checks
                              << " | Sample: " << sample
                              << " | Analytical: " << analytical
                              << " | Numerical: " << numerical << '\n';
                }

                if (std::isfinite(difference)
                    && difference > largest_difference) {
                    largest_difference = difference;
                }
            };

            for (std::size_t neuron = 0; neuron < output_weights.size(); ++neuron) {
                check(
                    output_weights[neuron],
                    gradients.output_weights[neuron]
                );
            }
            check(output_bias, gradients.output_bias);

            for (std::size_t neuron = 0; neuron < hidden_weights.size(); ++neuron) {
                for (std::size_t i = 0; i < input_count; ++i) {
                    check(
                        hidden_weights[neuron][i],
                        gradients.hidden_weights[neuron][i]
                    );
                }

                check(hidden_biases[neuron], gradients.hidden_biases[neuron]);
            }
        }

        std::cout << "Gradient checks: " << checks
                  << " | Failures: " << failures
                  << " | Largest finite difference: "
                  << largest_difference << '\n';

        return failures == 0;
    }
};

int main() {
    std::vector<std::vector<double>> training_inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<double> targets = {0.0, 1.0, 1.0, 0.0};

    // Two inputs, four hidden neurons, one output, seed 42.
    NeuralNetwork network(2, 4, 42);

    std::cout << "Trainable parameters: "
              << network.parameter_count() << '\n';

    if (!network.check_gradients(training_inputs, targets)) {
        std::cerr << "Gradient checks failed. Training stopped.\n";
        return 1;
    }

    double learning_rate = 0.1;
    double loss_threshold = 0.0001;
    int max_epochs = 100000;

    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        for (std::size_t sample = 0; sample < training_inputs.size(); ++sample) {
            network.train_one(
                training_inputs[sample], targets[sample], learning_rate
            );
        }

        double total_loss = 0.0;

        for (std::size_t sample = 0; sample < training_inputs.size(); ++sample) {
            total_loss += network.loss(
                training_inputs[sample], targets[sample]
            );
        }

        double average_loss = total_loss / training_inputs.size();

        if (!std::isfinite(average_loss)) {
            std::cerr << "Non-finite loss detected. Training stopped.\n";
            return 1;
        }

        if (epoch % 1000 == 0 || average_loss < loss_threshold
            || epoch == max_epochs - 1) {
            std::cout << "Epoch: " << epoch + 1
                      << " | Average loss: " << average_loss << '\n';
        }

        if (average_loss < loss_threshold) {
            std::cout << "Loss target reached.\n";
            break;
        }

        if (epoch == max_epochs - 1) {
            std::cout << "Epoch limit reached.\n";
        }
    }

    std::size_t correct = 0;
    std::cout << "\nFinal predictions:\n";

    for (std::size_t sample = 0; sample < training_inputs.size(); ++sample) {
        double output = network.predict(training_inputs[sample]);

        int predicted = 0;
        if (output >= 0.5) {
            predicted = 1;
        }

        if (predicted == targets[sample]) {
            ++correct;
        }

        std::cout << training_inputs[sample][0] << " XOR "
                  << training_inputs[sample][1]
                  << " | Output: " << output
                  << " | Predicted: " << predicted
                  << " | Target: " << targets[sample] << '\n';
    }

    double accuracy = 100.0 * correct / training_inputs.size();
    std::cout << "Training accuracy: " << accuracy << "%\n";

    if (correct != training_inputs.size()) {
        std::cout << "The network has not classified all XOR examples correctly.\n";
        return 1;
    }

    return 0;
}
