import math as m
import numpy as np
import pygad
import pygad.kerasga
from datetime import datetime

from snake_game import SnakeGame
from helper import Helper
from neural_network import NeuralNetwork

# Change OPTIMIZATION to True if you want to optimize params
OPTIMIZATION = False


class Optimization:
    def __init__(self):
        pass

    def save_logs(self, param, score):
        with open('logs/scores_' + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '.txt', 'a') as f:
            f.write(
                str('no_of_layers{}_no_of_neurons{}_snake_lr{}_score{}'.format(
                    int(
                        param['no_of_layers']),
                    param['no_of_neurons'],
                    param['lr'],
                    score)) + '\n')
            f.write('Params: ' + str(param) + '\n')


class Agent:
    def __init__(self, vectors_and_keys, n_games, total_score):
        self.vectors_and_keys = vectors_and_keys
        self.n_games = n_games
        self.total_score = total_score

    def generate_observation(self, snake, food, length, game):
        snake_direction_vector = self.get_snake_direction_vector(snake, length)
        food_direction_vector = self.get_food_direction_vector(
            snake, food, length)
        obstacle_front = self.get_obstacles(
            snake, snake_direction_vector, length, game)
        obstacle_right = self.get_obstacles(
            snake, self.turn_vector_to_the_right(snake_direction_vector), length, game)
        obstacle_left = self.get_obstacles(
            snake, self.turn_vector_to_the_left(snake_direction_vector), length, game)
        angle, snake_direction_vector_normalized, food_direction_vector_normalized = self.get_angle(
            snake_direction_vector, food_direction_vector, game)

        return np.array(
            [int(obstacle_front), int(obstacle_right), int(obstacle_left), snake_direction_vector_normalized[0],
             food_direction_vector_normalized[0], snake_direction_vector_normalized[1],
             food_direction_vector_normalized[1], angle])

    def get_snake_direction_vector(self, snake, length):
        return np.array(snake[length - 1]) - np.array(snake[length - 2])

    def get_food_direction_vector(self, snake, food, length):
        return np.array(food) - np.array(snake[length - 1])

    def get_obstacles(self, snake, snake_direction_vector, length, game):
        point = np.array(snake[length - 1]) + np.array(snake_direction_vector)

        return point.tolist() in snake[:-1] or point[0] < 0 or point[1] < 0 or point[0] >= game.DISPLAY_WIDHT or point[
            1] >= game.DISPLAY_HEIGHT

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, snake_direction, food_direction, game):
        norm_of_snake_direction_vector = np.linalg.norm(snake_direction)
        norm_of_food_direction_vector = np.linalg.norm(food_direction)

        if norm_of_snake_direction_vector == 0:
            norm_of_snake_direction_vector = game.SNAKE_BLOCK

        if norm_of_food_direction_vector == 0:
            norm_of_food_direction_vector = game.SNAKE_BLOCK

        snake_direction_vector_normalized = snake_direction / norm_of_snake_direction_vector
        food_direction_vector_normalized = food_direction / norm_of_food_direction_vector
        angle = m.atan2(food_direction_vector_normalized[1] * snake_direction_vector_normalized[0] -
                        food_direction_vector_normalized[0] * snake_direction_vector_normalized[1],
                        food_direction_vector_normalized[1] * snake_direction_vector_normalized[1] +
                        food_direction_vector_normalized[0] * snake_direction_vector_normalized[0]) / m.pi

        return angle, snake_direction_vector_normalized, food_direction_vector_normalized

    def get_game_action(self, snake, action, length):
        snake_direction_vector = self.get_snake_direction_vector(snake, length)
        new_direction = snake_direction_vector

        if action == -1:
            new_direction = self.turn_vector_to_the_left(
                snake_direction_vector)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(
                snake_direction_vector)

        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]

                return game_action

    def run_game_with_ML(self, game, model, helper):
        global observation

        max_score = 0
        final_score = 0

        game.reset()
        done, _, food, snake, length = game.generate_observations()

        count_same_direction = 0
        prev_direction = 0

        while game.MAX_STEPS != 0:
            observation = self.generate_observation(
                snake, food, length, game)

            predicted_direction = np.argmax(
                np.array(model.predict(np.array(observation).reshape(-1, 8))))

            if predicted_direction == prev_direction:
                count_same_direction += 1
            else:
                count_same_direction = 0
                prev_direction = predicted_direction

            # game_action = self.get_game_action(
            #     snake, predicted_direction, length)

            if done:
                final_score += -150

                break
            else:
                final_score += 0

            done, new_score, food, snake, length = game.game_loop(
                predicted_direction)

            if new_score - 2 > max_score:
                max_score = new_score - 2

            if count_same_direction > 5 and predicted_direction != 0:
                final_score -= 1
            else:
                final_score += 2

        if new_score > game.RECORD:
            game.RECORD = new_score

        self.n_games += 1
        self.total_score += new_score

        print('Game: ', self.n_games, 'Score: ',
              new_score, 'Record: ', game.RECORD)
        # print('Previous observation: ', observation)
        # print('Total score: ', self.total_score)

        helper.write_result_to_list(self.n_games, new_score)

        return final_score + max_score * 5000

    def save_test_logs(self, start_time, record_score):
        with open('logs/test_' + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '.txt', 'a') as f:
            f.write(str('start_time{}_record_score{}_total_score{}'.format(
                start_time, record_score, self.total_score)) + '\n')
            f.write('Values: {start_time: ' + str(start_time) + ', record_score: ' + str(record_score) +
                    ', total_score: ' + str(self.total_score) + '}\n')


def fitness_func(solution, sol_idx):
    global keras_ga, game, agent, model, helper

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=solution)

    model.set_weights(weights=model_weights_matrix)

    solution_fitness = agent.run_game_with_ML(game, model, helper)

    return solution_fitness


def callback_generation(ga_instance):
    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solution()[1]))


if __name__ == '__main__':
    optimization = Optimization()
    game = SnakeGame()

    vectors_and_keys = [[[-game.SNAKE_BLOCK, 0], 0],  # LEFT
                        [[game.SNAKE_BLOCK, 0], 1],  # RIGHT
                        [[0, -game.SNAKE_BLOCK], 2],  # UP
                        [[0, game.SNAKE_BLOCK], 3]]  # DOWN
    n_games = 0
    total_score = 0

    agent = Agent(vectors_and_keys, n_games, total_score)
    helper = Helper()

    if OPTIMIZATION == True:
        param = {
            'no_of_layers': 2,
            'no_of_neurons': 128,
            'lr': 0.001
        }

        no_of_layers = param['no_of_layers']
        no_of_neurons = param['no_of_neurons']
        lr = param['lr']

        # Build the keras model using the sequentional API
        network = NeuralNetwork(no_of_layers, no_of_neurons, lr)
        model = network.model()

        # Create an instance of the pygad.kerasga.KerasGA class to build the initial population
        keras_ga = pygad.kerasga.KerasGA(model=model,
                                         num_solutions=50)

        # Prepare the PyGAD parameters
        num_generations = 1000  # Number of generations

        # Number of solutions to be selected as parents in the mating pool
        num_parents_mating = 12

        # Initial population of network weights
        initial_population = keras_ga.population_weights
        parent_selection_type = "sss"  # Type of parent selection
        crossover_type = "single_point"  # Type of the crossover operator
        mutation_type = "random"  # Type of the mutation operator

        # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists
        mutation_percent_genes = 10

        # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing
        keep_parents = -1

        # Create an instance of the pygad.GA class
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               initial_population=initial_population,
                               fitness_func=fitness_func,
                               parent_selection_type=parent_selection_type,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes,
                               keep_parents=keep_parents,
                               on_generation=callback_generation)

        # Start the genetic algorithm evolution
        ga_instance.run()

        helper.write_result_to_csv()
        optimization.save_logs(param, agent.total_score)

        # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations
        ga_instance.plot_result(
            title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

        # Returning the details of the best solution
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Fitness value of the best solution = {solution_fitness}".format(
            solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(
            solution_idx=solution_idx))

        # Fetch the parameters of the best solution
        best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                      weights_vector=solution)
        model.set_weights(best_solution_weights)

        network.save_weights()
    else:
        no_of_layers = 2
        no_of_neurons = 128
        lr = 0.001

        # Build the keras model using the sequentional API
        network = NeuralNetwork(no_of_layers, no_of_neurons, lr)
        model = network.model()
        network.load_weights_()

        start_time = str(datetime.now().strftime("%Y%m%d%H%M%S"))

        for _ in range(2500):
            agent.run_game_with_ML(game, model, helper)

        agent.save_test_logs(start_time, game.RECORD)
