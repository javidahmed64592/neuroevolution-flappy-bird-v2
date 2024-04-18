from __future__ import annotations

from typing import List

import numpy as np
import pygame
from genetic_algorithm.ga import Member  # type: ignore
from neural_network.math.matrix import Matrix  # type: ignore
from neural_network.neural_network import NeuralNetwork  # type: ignore
from numpy.typing import NDArray


class Bird(Member):
    """
    Bird to use in PhraseSolver app.
    """

    GRAV = 1
    LIFT = -20
    MIN_VELOCITY = -10
    X_LIM = 1000
    Y_LIM = 1000

    def __init__(self, x: int, y: int, size: int, hidden_layer_sizes: List[int]) -> None:
        """
        Initialise bird with a starting position, a width and a height.

        Parameters:
            x (int): x coordinate of bird's start position
            y (int): y coordinate of bird's start position
            size (int): size of bird
            hidden_layer_sizes (List[int]): Neural network hidden layer sizes
        """
        super().__init__()
        self._x = x
        self._start_x = x
        self._y = y
        self._start_y = y
        self._velocity = 0
        self._size = size
        self._nn = NeuralNetwork(len(self.nn_input), 2, hidden_layer_sizes)

        self._score = 0
        self._alive = True
        self._colour = np.random.randint(low=0, high=256, size=3)

    @property
    def nn_input(self) -> NDArray:
        return np.array([self._y / self.Y_LIM])

    @property
    def chromosome(self) -> List[List[Matrix]]:
        return [self._nn.weights, self._nn.bias]

    @chromosome.setter
    def chromosome(self, new_chromosome: List[List[Matrix]]) -> None:
        self._nn.weights = new_chromosome[0]
        self._nn.bias = new_chromosome[1]

    @property
    def fitness(self) -> int:
        return self._score**2

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self._x, self._y, self._size, self._size)

    @property
    def velocity(self) -> int:
        return self._velocity

    @velocity.setter
    def velocity(self, new_velocity: int) -> None:
        self._velocity = max(new_velocity, self.MIN_VELOCITY)

    @property
    def offscreen(self) -> bool:
        return (0 > self._y) or (self._y + self._size > self.Y_LIM)

    def crossover(self, parent_a: Bird, parent_b: Bird, mutation_rate: int) -> None:
        """
        Crossover the chromosomes of two parents to create a new chromosome.

        Parameters:
            parent_a (Member): Used to construct new chromosome
            parent_b (Member): Used to construct new chromosome
            mutation_rate (int): Probability for mutations to occur
        """
        _new_weights = []
        _new_biases = []

        _zipped_chromosomes = zip(
            parent_a.chromosome[0], parent_b.chromosome[0], parent_a.chromosome[1], parent_b.chromosome[1]
        )
        for pa_weights, pb_weights, pa_bias, pb_bias in _zipped_chromosomes:
            _new_weight = Matrix.average_matrix(pa_weights, pb_weights)
            _new_weight = Matrix.mutated_matrix(_new_weight, mutation_rate, self._nn.WEIGHTS_RANGE)
            _new_bias = Matrix.average_matrix(pa_bias, pb_bias)
            _new_bias = Matrix.mutated_matrix(_new_bias, mutation_rate, self._nn.BIAS_RANGE)

            _new_weights.append(_new_weight)
            _new_biases.append(_new_bias)

        self._new_chromosome = [_new_weights, _new_biases]
        self._colour = np.average([self._colour, parent_a._colour, parent_b._colour], axis=0, weights=[0.7, 0.15, 0.15])

    def reset(self):
        """
        Reset to start positions.
        """
        self.velocity = 0
        self._x = self._start_x
        self._y = self._start_y
        self._score = 0
        self._alive = True

    def jump(self) -> None:
        """
        Make bird 'jump' by accelerating upwards.
        """
        self.velocity += self.LIFT

    def move(self) -> None:
        """
        Update bird's position and velocity.
        """
        self.velocity += self.GRAV
        self._y += self.velocity

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw Bird on the display.

        Parameter:
            screen (Surface): Screen to draw bird to
        """
        if not self._alive:
            return
        pygame.draw.rect(screen, self._colour.tolist(), self.rect)

    def update(self):
        """
        Use neural network to determine whether or not bird should jump, and kill if it collides with a Pipe.
        """
        if not self._alive:
            return

        output = self._nn.feedforward(self.nn_input)

        if output[0] < output[1]:
            self.jump()

        self.move()

        if self.offscreen:
            self._alive = False
            return

        self._score += 1
