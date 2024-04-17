from __future__ import annotations

import sys
from typing import List

import pygame
from pygame.locals import QUIT

from flappy_bird.flappy_bird_ga import FlappyBirdGA
from flappy_bird.objects.bird import Bird
from flappy_bird.pg.app import App


class FlappyBirdApp(App):
    """
    Simple app to use genetic algorithms to solve an alphanumeric phrase.
    """

    def __init__(self, fps: int) -> None:
        """
        Initialise FlappyBirdGA.

        Parameters:
            mutation_rate (int): Population mutation rate
        """
        super().__init__(fps)
        self._lifetime: int

    @property
    def screen(self):
        return self._display_surf

    @classmethod
    def create_app(
        cls, name: str, width: int, height: int, fps: int, font: str, font_size: int, lifetime: int
    ) -> FlappyBirdApp:
        Bird.Y_LIM = height
        fba: FlappyBirdApp = super().create_app(name, width, height, fps, font, font_size)
        fba._lifetime = lifetime
        return fba

    def add_ga(
        self,
        population_size: int,
        mutation_rate: int,
        x: int,
        y: int,
        size: int,
        nn_layer_sizes: List[int],
        weights_range: List[float],
        bias_range: List[float],
    ) -> None:
        """
        Add genetic algorithm to app.

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
        self._ga = FlappyBirdGA.create(
            population_size, mutation_rate, x, y, size, nn_layer_sizes, weights_range, bias_range
        )

    def update(self) -> None:
        """
        Move all birds and analyse population.
        """
        self._ga._evaluate()

    def run(self) -> None:
        """
        Run the application and handle events.
        """
        _count = 0
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            self._display_surf.fill((0, 0, 0))

            if _count == self._lifetime:
                self._ga._analyse()
                self._ga._evolve()
                _count = 0

            _count += 1
            self.update()
            pygame.display.update()
            self._clock.tick(self._fps)
