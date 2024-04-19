from __future__ import annotations

from typing import List

import numpy as np
from genetic_algorithm.ga import GeneticAlgorithm  # type: ignore
from neural_network.neural_network import NeuralNetwork  # type: ignore

from flappy_bird.objects.bird import Bird


class FlappyBirdGA(GeneticAlgorithm):
    """
    Genetic algorithm for Flappy Bird training.
    """

    def __init__(self, birds: list[Bird], mutation_rate: float) -> None:
        """
        Initialise FlappyBirdGA with a mutation rate.

        Parameters:
            birds (list[Bird]): Population of Birds
            mutation_rate (float): Population mutation rate
        """
        super().__init__(birds, mutation_rate)

    @property
    def num_alive(self) -> int:
        _alive_array = np.array([_bird._alive for _bird in self._population._population])
        return int(np.sum(_alive_array))

    @classmethod
    def create(
        cls,
        population_size: int,
        mutation_rate: float,
        x: int,
        y: int,
        size: int,
        hidden_layer_sizes: List[int],
        weights_range: List[float],
        bias_range: List[float],
    ) -> FlappyBirdGA:
        """
        Create genetic algorithm and configure neural network.

        Parameters:
            population_size (int): Number of members in population
            mutation_rate (float): Mutation rate for members
            x (int): x coordinate of bird's start position
            y (int): y coordinate of bird's start position
            size (int): Size of bird
            hidden_layer_sizes (List[int]): Neural network hidden layer sizes
            weights_range (List[float]): Range for random weights
            bias_range (List[float]): Range for random bias

        Returns:
            flappy_bird (FlappyBirdGA): Flappy Bird app
        """
        NeuralNetwork.WEIGHTS_RANGE = weights_range
        NeuralNetwork.BIAS_RANGE = bias_range
        flappy_bird = cls([Bird(x, y, size, hidden_layer_sizes) for _ in range(population_size)], mutation_rate)
        return flappy_bird

    def reset(self) -> None:
        """
        Reset all Birds.
        """
        for _bird in self._population._population:
            _bird.reset()
