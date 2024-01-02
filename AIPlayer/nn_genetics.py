from numpy import exp, array, random, dot, delete
from neural_ntw import NeuralNetwork

class NNGenetics():
	def __init__(self):
		return

	def fitness(self, nn_scores: array, nn_ball_collisions: array, n_of_parents = 2):
		parents = []
		fitness_list = []

		for i in range(len(nn_scores)):
			fitness = nn_scores[i] + 5000*nn_ball_collisions[i]
			fitness_list.append(fitness)
		fitness_arr = array(fitness_list)
		parent_finess = []
		for i in range(n_of_parents):
			parents.append(fitness_arr.argmax())
			parent_finess.append(fitness_arr[parents[i]])
			fitness_arr[parents[i]] = 0
		
		# print(parents)
		# print(scores)

		return parents, parent_finess

	def crossover(self, parents: [NeuralNetwork]):
		n_of_parents = len(parents)
		child = parents[0]
		n_of_layers = len(child.layers)
		parent_id = 0

		# print("Parents:")
		# for i in range(n_of_parents):
		# 	parents[i].print_weights()

		for i in range(n_of_layers):
			parent_id = i % n_of_parents
			child.layers[i] = parents[parent_id].layers[i]
			parent_id += 1

		# print("Child:")
		# child.print_weights()
		return child

	def mutate(self, child, prob_mut):
		for idx, layer in enumerate(child.layers):
			for idy, weights in enumerate(layer.synaptic_weights):
				for idz, weight in enumerate(weights):
					random_number = random.random()
					if random_number < (prob_mut / 3):
						# print(f"old weight: {child.layers[idx].synaptic_weights[idy][idz]}")
						child.layers[idx].synaptic_weights[idy][idz] += (2 * random.random() - 1) / 1000
						# print(f"new weight: {child.layers[idx].synaptic_weights[idy][idz]}")
					elif random_number < prob_mut:
						child.layers[idx].synaptic_weights[idy][idz] *= (1 + (2 * random.random() - 1) / 100)
		return child