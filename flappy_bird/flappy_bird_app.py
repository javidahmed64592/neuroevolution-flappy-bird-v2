from __future__ import annotations

from typing import List, cast

import pygame

from flappy_bird.flappy_bird_ga import FlappyBirdGA
from flappy_bird.objects.bird import Bird
from flappy_bird.pg.app import App


class FlappyBirdApp(App):
    """
    This class creates a version of Flappy Bird and uses neuroevolution to train AI to play the game.
    """

    def __init__(self, fps: int) -> None:
        """
        Initialise FlappyBirdApp.

        Parameters:
            fps (int): Application FPS
        """
        super().__init__(fps)
        self._lifetime: int
        self._count = 0

    @property
    def screen(self) -> pygame.Surface:
        return self._display_surf

    @classmethod
    def create_game(
        cls, name: str, width: int, height: int, fps: int, font: str, font_size: int, lifetime: int
    ) -> FlappyBirdApp:
        """
        Create App and configure limits for Bird and genetic algorithm.

        Parameters:
            name (str): Application name
            width (int): Screen width
            height (int): Screen height
            fps (int): Application FPS
            font (str): Font style
            font_size (int): Font size
            lifetime (int): Length of each generation

        Returns:
            fba (FlappyBirdApp): Flappy Bird application
        """
        Bird.X_LIM = width
        Bird.Y_LIM = height
        fba = cast(FlappyBirdApp, super().create_app(name, width, height, fps, font, font_size))
        fba._lifetime = lifetime
        return fba

    def add_ga(
        self,
        population_size: int,
        mutation_rate: float,
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
            mutation_rate (float): Mutation rate for members
            x (int): x coordinate of bird's start position
            y (int): y coordinate of bird's start position
            size (int): Size of bird
            nn_layer_sizes (List[int]): Neural network layer sizes
            weights_range (List[float]): Range for random weights
            bias_range (List[float]): Range for random bias
        """
        self._ga = FlappyBirdGA.create(
            population_size, mutation_rate, x, y, size, nn_layer_sizes, weights_range, bias_range
        )

    def update(self) -> None:
        """
        Run genetic algorithm, update Birds and draw to screen.
        """
        if int(self._count) == (self._lifetime * self._fps) or self._ga.num_alive == 0:
            self._ga._analyse()
            self._ga._evolve()
            self._ga.reset()
            self._count = 0

        self._ga._evaluate(self.screen)
        self._count += 1
        self.write_stats()

    def write_stats(self) -> None:
        """
        Write algorithm statistics to screen.
        """
        start_x = 20
        start_y = 30
        self.write_text(f"Generation: {self._ga._generation}", start_x, start_y)
        self.write_text(f"Birds alive: {self._ga.num_alive}", start_x, start_y * 3)
        self.write_text(f"Score: {int(self._count / self._fps)}", start_x, start_y * 4)
