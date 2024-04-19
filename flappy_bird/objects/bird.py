from __future__ import annotations

import numpy as np
import pygame
from genetic_algorithm.ga import Member  # type: ignore
from neural_network.math.matrix import Matrix  # type: ignore
from neural_network.neural_network import NeuralNetwork  # type: ignore
from numpy.typing import NDArray

from flappy_bird.objects.pipe import Pipe


class Bird(Member):
    """
    Bird to use in Flappy Bird app.
    """

    GRAV = 1
    LIFT = -20
    X = 50
    MIN_VELOCITY = -10
    X_LIM = 1000
    Y_LIM = 1000

    def __init__(self, y: int, size: int, hidden_layer_sizes: list[int]) -> None:
        """
        Initialise Bird with a starting position, a width and a height.

        Parameters:
            y (int): y coordinate of Bird's start position
            size (int): Size of Bird
            hidden_layer_sizes (list[int]): Neural network hidden layer sizes
        """
        super().__init__()
        self._y = y
        self._start_y = y
        self._velocity = 0
        self._size = size
        self._nn = NeuralNetwork(len(self.nn_input), 2, hidden_layer_sizes)

        self._score = 0
        self._alive = True
        self._colour = np.random.randint(low=0, high=256, size=3)
        self._closest_pipe: Pipe

    @property
    def nn_input(self) -> NDArray:
        return np.array([self._y / self.Y_LIM])

    @property
    def chromosome(self) -> list[list[Matrix]]:
        return [self._nn.weights, self._nn.bias]

    @chromosome.setter
    def chromosome(self, new_chromosome: list[list[Matrix]]) -> None:
        self._nn.weights = new_chromosome[0]
        self._nn.bias = new_chromosome[1]

    @property
    def fitness(self) -> int:
        return self._score**2

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.X, self._y, self._size, self._size)

    @property
    def velocity(self) -> int:
        return self._velocity

    @velocity.setter
    def velocity(self, new_velocity: int) -> None:
        self._velocity = max(new_velocity, self.MIN_VELOCITY)

    @property
    def offscreen(self) -> bool:
        return (0 > self._y) or (self._y + self._size > self.Y_LIM)

    @property
    def collide_with_closest_pipe(self) -> bool:
        """
        Check if Bird is colliding with closest Pipe.

        Returns:
            (bool): Is bird colliding with pipe?
        """
        if not self._closest_pipe:
            return False
        return self.rect.colliderect(self._closest_pipe.rects[0]) or self.rect.colliderect(self._closest_pipe.rects[1])

    def crossover(self, parent_a: Bird, parent_b: Bird, mutation_rate: int) -> None:
        """
        Crossover the chromosomes of two Birds to create a new chromosome.

        Parameters:
            parent_a (Member): Used to construct new chromosome
            parent_b (Member): Used to construct new chromosome
            mutation_rate (int): Probability for mutations to occur
        """
        _new_weights = []
        _new_biases = []

        _zipped_chromosomes = zip(
            parent_a.chromosome[0],
            parent_b.chromosome[0],
            parent_a.chromosome[1],
            parent_b.chromosome[1],
            strict=False,
        )
        for pa_weights, pb_weights, pa_bias, pb_bias in _zipped_chromosomes:
            _new_weight = Matrix.average_matrix(pa_weights, pb_weights)
            _new_weight = Matrix.mutated_matrix(_new_weight, mutation_rate, self._nn.WEIGHTS_RANGE)
            _new_bias = Matrix.average_matrix(pa_bias, pb_bias)
            _new_bias = Matrix.mutated_matrix(_new_bias, mutation_rate, self._nn.BIAS_RANGE)

            _new_weights.append(_new_weight)
            _new_biases.append(_new_bias)

        self._new_chromosome = [_new_weights, _new_biases]
        self._colour = np.average(
            [self._colour, parent_a._colour, parent_b._colour],
            axis=0,
            weights=[0.98, 0.01, 0.01],
        )

    def reset(self) -> None:
        """
        Reset to start positions.
        """
        self.velocity = 0
        self._y = self._start_y
        self._score = 0
        self._alive = True

    def jump(self) -> None:
        """
        Make Bird 'jump' by accelerating upwards.
        """
        self.velocity += self.LIFT

    def move(self) -> None:
        """
        Update Bird's position and velocity.
        """
        self.velocity += self.GRAV
        self._y += self.velocity

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw Bird on the display.

        Parameters:
            screen (Surface): Screen to draw Bird to
        """
        if not self._alive:
            return
        pygame.draw.rect(screen, self._colour.tolist(), self.rect)

    def update(self, closest_pipe: Pipe) -> None:
        """
        Use neural network to determine whether or not Bird should jump, and kill if it collides with a Pipe.

        Parameters:
            closest_pipe (Pipe): Pipe closest to Bird
        """
        if not self._alive:
            return

        self._closest_pipe = closest_pipe
        output = self._nn.feedforward(self.nn_input)

        if output[0] < output[1]:
            self.jump()

        self.move()

        if self.offscreen or self.collide_with_closest_pipe:
            self._alive = False
            return

        self._score += 1
