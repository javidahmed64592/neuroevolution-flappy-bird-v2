from __future__ import annotations

from typing import List

import numpy as np
from genetic_algorithm.ga import Member
from neural_network.math.matrix import Matrix
from neural_network.neural_network import NeuralNetwork
from numpy.typing import NDArray


class Bird(Member):
    """
    Bird to use in PhraseSolver app.
    """

    GRAV = 1
    LIFT = -20
    MIN_VELOCITY = -10
    Y_LIM = 1000

    def __init__(self, x: int, y: int, size: int, nn_layer_sizes: List[int]) -> None:
        """
        Initialise bird with a starting position, a width and a height.

        Parameters:
            x (int): x coordinate of bird's start position
            y (int): y coordinate of bird's start position
            size (int): size of bird
            nn_layer_sizes (List[int]): Neural network layer sizes
        """
        super().__init__()
        self._x = x
        self._start_x = x
        self._y = y
        self._start_y = y
        self._velocity = 0

        self._size = size
        self._nn_layer_sizes = nn_layer_sizes
        self._nn = NeuralNetwork(
            num_inputs=self.num_inputs, num_outputs=self.num_outputs, hidden_layer_sizes=self.hidden_layer_sizes
        )

        self._score = 0
        self._alive = True

    @property
    def num_inputs(self) -> int:
        return self._nn_layer_sizes[0]

    @property
    def num_outputs(self) -> int:
        return self._nn_layer_sizes[-1]

    @property
    def hidden_layer_sizes(self) -> List[int]:
        return self._nn_layer_sizes[1:-1]

    @property
    def chromosome(self) -> List[List[Matrix]]:
        return [self._nn.weights, self._nn.bias]

    @chromosome.setter
    def chromosome(self, new_chromosome: List[Matrix]) -> None:
        self._nn.weights = new_chromosome[0]
        self._nn.bias = new_chromosome[1]

    @property
    def fitness(self) -> int:
        return self._score**2

    @property
    def nn_input(self) -> NDArray:
        return np.array([self._y / self.Y_LIM])

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

        self._new_chromosome = [_new_weights, _new_bias]

    def reset(self):
        """
        Reset to start positions.
        """
        self._velocity = 0
        self._x = self._start_x
        self._y = self._start_y
        self._score = 0
        self._alive = True

    def jump(self) -> None:
        """
        Make bird 'jump' by accelerating upwards.
        """
        self._velocity += self.LIFT

    def move(self) -> None:
        """
        Update bird's position and velocity.
        """
        self._velocity += self.GRAV
        self._velocity = max(self._velocity, self.MIN_VELOCITY)
        self._y += self._velocity

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

        # If bird collides with pipe, kill and return
        if self.offscreen:
            self._alive = False
            return

        self._score += 1
