from __future__ import annotations

from typing import List

import numpy as np
from genetic_algorithm.ga import GeneticAlgorithm
from neural_network.neural_network import NeuralNetwork

from flappy_bird.objects.bird import Bird


class FlappyBirdGA(GeneticAlgorithm):
    """
    Simple app to use genetic algorithms to solve an alphanumeric phrase.
    """

    def __init__(self, mutation_rate: int) -> None:
        """
        Initialise FlappyBirdGA.

        Parameters:
            mutation_rate (int): Population mutation rate
        """
        super().__init__(mutation_rate)

    @classmethod
    def create(
        cls,
        population_size: int,
        mutation_rate: int,
        x: int,
        y: int,
        size: int,
        nn_layer_sizes: List[int],
        weights_range: List[float],
        bias_range: List[float],
    ) -> FlappyBirdGA:
        """
        Create app with game and genetic algorithm variables.

        Parameters:
            population_size (int): Number of members in population
            mutation_rate (int): Mutation rate for members
            x (int): x coordinate of bird's start position
            y (int): y coordinate of bird's start position
            size (int): Size of bird
            nn_layer_sizes (List[int]): Neural network layer sizes
            weights_range (List[float]): Range for random weights
            bias_range (List[float]): Range for random bias

        Returns:
            flappy_bird (FlappyBirdGA): Flappy Bird app
        """
        flappy_bird = cls(mutation_rate)
        NeuralNetwork.WEIGHTS_RANGE = weights_range
        NeuralNetwork.BIAS_RANGE = bias_range
        flappy_bird._add_population([Bird(x, y, size, nn_layer_sizes) for _ in range(population_size)])
        return flappy_bird

    def _evaluate(self) -> None:
        """
        Evaluate the population.
        """
        for _bird in self._population._population:
            _bird.update()
        self._population.evaluate()

    def _analyse(self) -> None:
        """
        Analyse best member's chromosome.
        """
        _gen_text = f"Generation {self._generation:>4}:"
        _max_fitness_text = f"Max Fitness: {self._population.best_fitness}"
        _avg_fitness_text = f"Average Fitness: {np.average(self._population._population_fitness)}"
        print(f"{_gen_text} \t{_max_fitness_text} \t{_avg_fitness_text}")