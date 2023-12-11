from numpy import exp, array, random, dot, savetxt, loadtxt, append, vstack, hstack

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs_per_neuron = number_of_inputs_per_neuron
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, num_of_inputs, num_of_outputs, hidden_layers):
        self.num_of_inputs = num_of_inputs
        self.num_of_outputs = num_of_outputs
        self.hidden_layers = hidden_layers
        self.layers = []
        for layer in range(len(hidden_layers)+1):
            if layer == 0:
                self.layers.append(NeuronLayer(hidden_layers[layer], num_of_inputs))
            elif layer == len(hidden_layers):
                self.layers.append(NeuronLayer(num_of_outputs, hidden_layers[layer-1]))
            else:
                self.layers.append(NeuronLayer(hidden_layers[layer], hidden_layers[layer-1]))

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        outputs = []
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                output = self.__sigmoid(dot(inputs, layer.synaptic_weights))
            else:
                output = self.__sigmoid(dot(outputs[idx-1], layer.synaptic_weights))
            outputs.append(output)
        # output_from_layer1 = self.__sigmoid(dot(inputs, self.layers[0].synaptic_weights))
        # output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layers[1].synaptic_weights))

        return outputs

    # The neural network prints its weights
    def print_weights(self):
        for idx, layer in enumerate(self.layers):
            print(f"    Layer {idx} ({layer.number_of_neurons} neurons, each with {layer.number_of_inputs_per_neuron} inputs): ")
            print(layer.synaptic_weights)

    # The neural network prints its weights
    def save_weights(self, filename):
        # self.print_weights()
        f=open(filename,'a')
        for idx, layer in enumerate(self.layers):
            savetxt(f, layer.synaptic_weights, delimiter=",")
        f.close()

    def load_weights(self, filename):
        f=open(filename,'r')
        # lines = loadtxt(filename, comments="#", delimiter=",", unpack=False)
        # print(lines)
        for idx, layer in enumerate(self.layers):
            layer.synaptic_weights = array([])
            for neuron in range(layer.number_of_inputs_per_neuron):
                line = f.readline()
                # print(line)
                line_arr = array(line.split(',')).astype(float)
                # print(neuron)
                # print(line_arr)
                if(neuron == 0):
                    layer.synaptic_weights = hstack((layer.synaptic_weights, line_arr))
                else:
                    layer.synaptic_weights = vstack((layer.synaptic_weights, line_arr))
            # print(f"    Layer {idx} ({layer.number_of_neurons} neurons, each with {layer.number_of_inputs_per_neuron} inputs): ")
            # print(layer.synaptic_weights)
        f.close()
        self.print_weights()

    # The neural network prints its outputs
    def print_outputs(self, outputs):
        for idx, layer in enumerate(self.layers):
            print(f"    Layer {idx} ({layer.number_of_neurons} neurons) outputs: ")
            print(outputs[idx])