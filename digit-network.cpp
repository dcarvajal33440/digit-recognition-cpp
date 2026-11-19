#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

class NeuralNetwork {
private:
    std::size_t input_count;
    std::size_t hidden_count;
    std::size_t output_count;

    // Each row contains the incoming weights for one neuron.
    std::vector<std::vector<double>> hidden_weights;
    std::vector<double> hidden_biases;

    std::vector<std::vector<double>> output_weights;
    std::vector<double> output_biases;

    struct ForwardPass {
        std::vector<double> hidden;
        std::vector<double> logits;
        std::vector<double> probabilities;
    };

    struct Gradients {
        std::vector<std::vector<double>> hidden_weights;
        std::vector<double> hidden_biases;
        std::vector<std::vector<double>> output_weights;
        std::vector<double> output_biases;
    };

    static double sigmoid(double value) {
        if (value >= 0.0) {
            return 1.0 / (1.0 + std::exp(-value));
        }

        double exponential = std::exp(value);
        return exponential / (1.0 + exponential);
    }

    void validate_inputs(const std::vector<double>& inputs) const {
        if (inputs.size() != input_count) {
            throw std::invalid_argument("Incorrect number of inputs.");
        }

        for (double value : inputs) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Inputs must be finite.");
            }
        }
    }

    void validate_target(std::size_t target) const {
        if (target >= output_count) {
            throw std::invalid_argument("Target class is out of range.");
        }
    }

    ForwardPass forward(const std::vector<double>& inputs) const {
        validate_inputs(inputs);

        ForwardPass result;
        result.hidden.resize(hidden_count);
        result.logits.resize(output_count);
        result.probabilities.resize(output_count);

        // Input layer -> sigmoid hidden layer.
        for (std::size_t h = 0; h < hidden_count; ++h) {
            double wsum = hidden_biases[h];

            for (std::size_t i = 0; i < input_count; ++i) {
                wsum += hidden_weights[h][i] * inputs[i];
            }

            result.hidden[h] = sigmoid(wsum);
        }

        // Hidden layer -> raw output scores, called logits.
        for (std::size_t o = 0; o < output_count; ++o) {
            double wsum = output_biases[o];

            for (std::size_t h = 0; h < hidden_count; ++h) {
                wsum += output_weights[o][h] * result.hidden[h];
            }

            if (!std::isfinite(wsum)) {
                throw std::runtime_error("Non-finite output score.");
            }

            result.logits[o] = wsum;
        }

        // Stable softmax: subtract the largest score before exp().
        double largest = *std::max_element(
            result.logits.begin(), result.logits.end()
        );

        double total = 0.0;

        for (std::size_t o = 0; o < output_count; ++o) {
            result.probabilities[o] =
                std::exp(result.logits[o] - largest);
            total += result.probabilities[o];
        }

        for (std::size_t o = 0; o < output_count; ++o) {
            result.probabilities[o] /= total;
        }

        return result;
    }

    Gradients calculate_gradients(
        const std::vector<double>& inputs,
        std::size_t target
    ) const {
        validate_target(target);
        ForwardPass result = forward(inputs);

        Gradients gradients;
        gradients.hidden_weights =
            std::vector<std::vector<double>>(
                hidden_count, std::vector<double>(input_count, 0.0)
            );
        gradients.hidden_biases =
            std::vector<double>(hidden_count, 0.0);
        gradients.output_weights =
            std::vector<std::vector<double>>(
                output_count, std::vector<double>(hidden_count, 0.0)
            );
        gradients.output_biases =
            std::vector<double>(output_count, 0.0);

        // Softmax + cross-entropy derivative: probability - target.
        std::vector<double> output_gradients = result.probabilities;
        output_gradients[target] -= 1.0;

        for (std::size_t o = 0; o < output_count; ++o) {
            gradients.output_biases[o] = output_gradients[o];

            for (std::size_t h = 0; h < hidden_count; ++h) {
                gradients.output_weights[o][h] =
                    output_gradients[o] * result.hidden[h];
            }
        }

        // Each hidden neuron affects every output neuron.
        for (std::size_t h = 0; h < hidden_count; ++h) {
            double propagated_gradient = 0.0;

            for (std::size_t o = 0; o < output_count; ++o) {
                propagated_gradient +=
                    output_gradients[o] * output_weights[o][h];
            }

            double activation = result.hidden[h];
            double hidden_gradient =
                propagated_gradient * activation * (1.0 - activation);

            gradients.hidden_biases[h] = hidden_gradient;

            for (std::size_t i = 0; i < input_count; ++i) {
                gradients.hidden_weights[h][i] =
                    hidden_gradient * inputs[i];
            }
        }

        return gradients;
    }

    double numerical_gradient(
        double& parameter,
        const std::vector<double>& inputs,
        std::size_t target
    ) {
        double epsilon = 0.00001;
        double original = parameter;
        double loss_plus;
        double loss_minus;

        // Restore the parameter even if loss() throws an exception.
        try {
            parameter = original + epsilon;
            loss_plus = loss(inputs, target);

            parameter = original - epsilon;
            loss_minus = loss(inputs, target);
        } catch (...) {
            parameter = original;
            throw;
        }

        parameter = original;
        return (loss_plus - loss_minus) / (2.0 * epsilon);
    }

public:
    NeuralNetwork(
        std::size_t inputs,
        std::size_t hidden_neurons,
        std::size_t outputs,
        unsigned int seed
    )
        : input_count(inputs),
          hidden_count(hidden_neurons),
          output_count(outputs),
          hidden_weights(
              hidden_neurons, std::vector<double>(inputs, 0.0)
          ),
          hidden_biases(hidden_neurons, 0.0),
          output_weights(
              outputs, std::vector<double>(hidden_neurons, 0.0)
          ),
          output_biases(outputs, 0.0) {
        if (inputs == 0 || hidden_neurons == 0 || outputs < 2) {
            throw std::invalid_argument(
                "Use positive input/hidden sizes and at least two outputs."
            );
        }

        std::mt19937 generator(seed);

        double hidden_limit = std::sqrt(
            6.0 / (static_cast<double>(inputs) + hidden_neurons)
        );
        double output_limit = std::sqrt(
            6.0 / (static_cast<double>(hidden_neurons) + outputs)
        );

        std::uniform_real_distribution<double> hidden_distribution(
            -hidden_limit, hidden_limit
        );
        std::uniform_real_distribution<double> output_distribution(
            -output_limit, output_limit
        );

        for (std::size_t h = 0; h < hidden_count; ++h) {
            for (std::size_t i = 0; i < input_count; ++i) {
                hidden_weights[h][i] = hidden_distribution(generator);
            }
        }

        for (std::size_t o = 0; o < output_count; ++o) {
            for (std::size_t h = 0; h < hidden_count; ++h) {
                output_weights[o][h] = output_distribution(generator);
            }
        }
    }

    std::size_t parameter_count() const {
        return hidden_count * input_count + hidden_count
             + output_count * hidden_count + output_count;
    }

    std::vector<double> predict(
        const std::vector<double>& inputs
    ) const {
        return forward(inputs).probabilities;
    }

    std::size_t classify(const std::vector<double>& inputs) const {
        std::vector<double> probabilities = predict(inputs);
        std::size_t best = 0;

        for (std::size_t o = 1; o < output_count; ++o) {
            if (probabilities[o] > probabilities[best]) {
                best = o;
            }
        }

        return best;
    }

    double loss(
        const std::vector<double>& inputs,
        std::size_t target
    ) const {
        validate_target(target);
        ForwardPass result = forward(inputs);

        double largest = *std::max_element(
            result.logits.begin(), result.logits.end()
        );

        double sum_exp = 0.0;
        for (double logit : result.logits) {
            sum_exp += std::exp(logit - largest);
        }

        // Stable equivalent of -log(probability of the correct class).
        return (largest - result.logits[target]) + std::log(sum_exp);
    }

    void train_one(
        const std::vector<double>& inputs,
        std::size_t target,
        double learning_rate
    ) {
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0) {
            throw std::invalid_argument("Learning rate must be positive.");
        }

        Gradients gradients = calculate_gradients(inputs, target);

        for (std::size_t o = 0; o < output_count; ++o) {
            for (std::size_t h = 0; h < hidden_count; ++h) {
                output_weights[o][h] -=
                    learning_rate * gradients.output_weights[o][h];
            }

            output_biases[o] -=
                learning_rate * gradients.output_biases[o];
        }

        for (std::size_t h = 0; h < hidden_count; ++h) {
            for (std::size_t i = 0; i < input_count; ++i) {
                hidden_weights[h][i] -=
                    learning_rate * gradients.hidden_weights[h][i];
            }

            hidden_biases[h] -=
                learning_rate * gradients.hidden_biases[h];
        }
    }

    bool check_gradients(
        const std::vector<double>& inputs,
        std::size_t target
    ) {
        Gradients gradients = calculate_gradients(inputs, target);

        std::size_t checks = 0;
        std::size_t failures = 0;
        double largest_difference = 0.0;

        auto check = [&](double& parameter, double analytical) {
            double numerical = numerical_gradient(parameter, inputs, target);
            double difference = std::abs(analytical - numerical);

            double tolerance = 1e-7 + 1e-5 *
                std::max(std::abs(analytical), std::abs(numerical));

            ++checks;

            if (!std::isfinite(difference) || difference > tolerance) {
                ++failures;
                std::cout << "Failed check " << checks
                          << " | Analytical: " << analytical
                          << " | Numerical: " << numerical << '\n';
            }

            if (std::isfinite(difference)) {
                largest_difference = std::max(
                    largest_difference, difference
                );
            }
        };

        for (std::size_t o = 0; o < output_count; ++o) {
            for (std::size_t h = 0; h < hidden_count; ++h) {
                check(output_weights[o][h], gradients.output_weights[o][h]);
            }
            check(output_biases[o], gradients.output_biases[o]);
        }

        for (std::size_t h = 0; h < hidden_count; ++h) {
            for (std::size_t i = 0; i < input_count; ++i) {
                check(hidden_weights[h][i], gradients.hidden_weights[h][i]);
            }
            check(hidden_biases[h], gradients.hidden_biases[h]);
        }

        std::cout << "Gradient checks: " << checks
                  << " | Failures: " << failures
                  << " | Largest finite difference: "
                  << largest_difference << '\n';

        return failures == 0;
    }
};

int main() {
    try {
        // A tiny network makes exhaustive gradient checking inexpensive.
        NeuralNetwork test_network(2, 3, 3, 42);

        std::cout << "Checking the multiclass derivatives:\n";

        for (std::size_t target = 0; target < 3; ++target) {
            if (!test_network.check_gradients({0.3, 0.7}, target)) {
                std::cerr << "Gradient checking failed.\n";
                return 1;
            }
        }

        // Three simple classes to verify that training reduces loss.
        std::vector<std::vector<double>> training_inputs = {
            {0.0, 0.0},
            {0.0, 0.2},
            {1.0, 0.0},
            {0.8, 0.1},
            {0.0, 1.0},
            {0.1, 0.8}
        };
        std::vector<std::size_t> targets = {0, 0, 1, 1, 2, 2};

        double learning_rate = 0.1;
        double loss_threshold = 0.01;
        int max_epochs = 10000;

        for (int epoch = 0; epoch < max_epochs; ++epoch) {
            for (std::size_t sample = 0; sample < training_inputs.size(); ++sample) {
                test_network.train_one(
                    training_inputs[sample], targets[sample], learning_rate
                );
            }

            double total_loss = 0.0;
            for (std::size_t sample = 0; sample < training_inputs.size(); ++sample) {
                total_loss += test_network.loss(
                    training_inputs[sample], targets[sample]
                );
            }

            double average_loss = total_loss / training_inputs.size();

            if (!std::isfinite(average_loss)) {
                throw std::runtime_error("Training loss is not finite.");
            }

            if (epoch % 500 == 0 || average_loss < loss_threshold
                || epoch == max_epochs - 1) {
                std::cout << "Epoch: " << epoch + 1
                          << " | Average loss: " << average_loss << '\n';
            }

            if (average_loss < loss_threshold) {
                std::cout << "Loss target reached.\n";
                break;
            }
        }

        std::size_t correct = 0;
        std::cout << "\nToy dataset predictions:\n";

        for (std::size_t sample = 0; sample < training_inputs.size(); ++sample) {
            std::size_t predicted =
                test_network.classify(training_inputs[sample]);

            std::cout << "Sample: " << sample
                      << " | Predicted: " << predicted
                      << " | Target: " << targets[sample] << '\n';

            if (predicted == targets[sample]) {
                ++correct;
            }
        }

        std::cout << "Toy training accuracy: "
                  << 100.0 * correct / training_inputs.size() << "%\n";

        if (correct != training_inputs.size()) {
            std::cerr << "Toy learning check did not reach 100% accuracy.\n";
            return 1;
        }

        // Our digit model: initialized, but not yet trained.
        NeuralNetwork digit_network(784, 64, 10, 42);

        std::cout << "\nDigit network: 784 -> 64 -> 10\n";
        std::cout << "Trainable parameters: "
                  << digit_network.parameter_count() << '\n';
        std::cout << "Ready for MNIST loading and training.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
