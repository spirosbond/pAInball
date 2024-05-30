import numpy as np
from numpy import exp, array, random, dot, delete, savetxt
from neural_ntw import NeuralNetwork
import csv
import copy


class NNGenetics:
    def __init__(self):
        return

    def fitness(
        self,
        nn_scores: array,
        nn_ball_collisions_l: array,
        nn_ball_collisions_r: array,
        game_durations: array,
        balls_used: array,
        n_of_parents=2,
    ):
        parents = []
        fitness_list = []
        print("Starting fitness")
        for i in range(len(nn_scores)):
            # print(f"fitness: {i}")
            fitness = (
                nn_scores[i]
                + 10000 * (nn_ball_collisions_l[i] + nn_ball_collisions_r[i])
                - balls_used[i] * 50000
            )
            if nn_ball_collisions_l[i] > 0 and nn_ball_collisions_r[i] > 0:
                fitness *= 2
            fitness /= game_durations[i]
            fitness_list.append(fitness)
        # print("Calculated fitnesses")
        fitness_arr = array(fitness_list)
        parent_fitness = []
        for i in range(n_of_parents):
            parents.append(fitness_arr.argmax())
            parent_fitness.append(fitness_arr[parents[i]])
            fitness_arr[parents[i]] = 0
        print("Ranked fitnesses")
        # print(parents)
        # print(parent_fitness)

        return parents, parent_fitness

    def save_fitness(self, epoch, parent_ids, parent_fitness, filename):
        row_data = [epoch] + parent_fitness + parent_ids
        np_row_data = np.array([row_data])
        # for idx, parent in enumerate(parent_ids):
        with open(filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(np_row_data)

    def crossover_shuffle(self, parents: [NeuralNetwork]):
        n_of_parents = len(parents)
        child = copy.deepcopy(parents[0])
        n_of_layers = len(child.layers)
        parent_id = 0

        # print("Parents:")
        # for i in range(n_of_parents):
        #     parents[i].print_weights()

        for i in range(n_of_layers):
            parent_id = i % n_of_parents
            child.layers[i] = parents[parent_id].layers[i]
            parent_id += 1

        # print("Child:")
        # child.print_weights()
        return child

    def crossover_average(self, parents: [NeuralNetwork]):
        n_of_parents = len(parents)
        child = copy.deepcopy(parents[0])
        n_of_layers = len(child.layers)

        # print("Parents:")
        # for i in range(n_of_parents):
        #     parents[i].print_weights()

        for i in range(n_of_layers):
            avg_layer_weights = np.zeros(
                np.shape(parents[0].layers[i].synaptic_weights)
            )
            for j in range(n_of_parents):
                avg_layer_weights += np.array(parents[j].layers[i].synaptic_weights)
            avg_layer_weights = avg_layer_weights / n_of_parents
            # parent_id = i % n_of_parents
            child.layers[i].synaptic_weights = avg_layer_weights

        # print("Child:")
        # child.print_weights()
        return child

    def mutate(self, child, prob_mut):
        mutant = copy.deepcopy(child)
        for idx, layer in enumerate(child.layers):
            for idy, weights in enumerate(layer.synaptic_weights):
                for idz, weight in enumerate(weights):
                    random_number = random.random()
                    if random_number < (prob_mut / 3):
                        # print(f"old weight: {child.layers[idx].synaptic_weights[idy][idz]}")
                        mutant.layers[idx].synaptic_weights[idy][idz] += (
                            random.uniform(-1, 1) / 10
                        )
                        # print(f"new weight: {child.layers[idx].synaptic_weights[idy][idz]}")
                    elif random_number < prob_mut:
                        mutant.layers[idx].synaptic_weights[idy][idz] *= (
                            1 + random.uniform(-1, 1) / 10
                        )
        return mutant
