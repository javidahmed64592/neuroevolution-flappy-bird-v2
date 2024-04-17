from __future__ import annotations

from typing import List

import numpy as np
from genetic_algorithm.ga import GeneticAlgorithm
from neural_network.neural_network import NeuralNetwork

from flappy_bird.objects.bird import Bird
from flappy_bird.pg.app import App


class FlappyBirdApp(GeneticAlgorithm):
    """
    Simple app to use genetic algorithms to solve an alphanumeric phrase.
    """

    def __init__(self, mutation_rate: int) -> None:
        """
        Initialise PhraseSolver app.

        Parameters:
            mutation_rate (int)
        """
        super().__init__(mutation_rate)
        self._lifetime: int

    @classmethod
    def create_and_run(
        cls,
        population_size: int,
        mutation_rate: int,
        x: int,
        y: int,
        size: int,
        nn_layer_sizes: List[int],
        weights_range: List[float],
        bias_range: List[float],
        lifetime: int,
    ) -> FlappyBirdApp:
        """
        Create app and run genetic algorithm.

        Parameters:
            population_size (int): Number of members in population
            mutation_rate (int): Mutation rate for members
            x (int): x coordinate of bird's start position
            y (int): y coordinate of bird's start position
            size (int): Size of bird
            nn_layer_sizes (List[int]): Neural network layer sizes
            weights_range (List[float]): Range for random weights
            bias_range (List[float]): Range for random bias
            lifetime (int): Number of iterations before

        Returns:
            ga (PhraseSolver): Phrase solver app
        """
        ga = cls(mutation_rate)
        NeuralNetwork.WEIGHTS_RANGE = weights_range
        NeuralNetwork.BIAS_RANGE = bias_range
        ga._add_population([Bird(x, y, size, nn_layer_sizes) for _ in range(population_size)])
        ga._lifetime = lifetime
        ga.run()
        return ga

    def _evaluate(self) -> None:
        """
        Evaluate the population.
        """
        for _ in range(self._lifetime):
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
