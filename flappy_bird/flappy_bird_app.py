from __future__ import annotations

from typing import List, cast

from flappy_bird.flappy_bird_ga import FlappyBirdGA
from flappy_bird.objects.bird import Bird
from flappy_bird.pg.app import App


class FlappyBirdApp(App):
    """
    This class creates a version of Flappy Bird and uses neuroevolution to train AI to play the game.
    """

    def __init__(self, name: str, width: int, height: int, fps: int, font: str, font_size: int) -> None:
        """
        Initialise FlappyBirdApp.

        Parameters:
            name (str): App name
            width (int): Screen width
            height (int): Screen height
            fps (int): Game FPS
            font (str): Font style
            font_size (int): Font size
        """
        super().__init__(name, width, height, fps, font, font_size)
        self._lifetime: int
        self._count = 0

    @property
    def max_count(self) -> int:
        return self._lifetime * self._fps

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

    def _write_stats(self) -> None:
        """
        Write algorithm statistics to screen.
        """
        _start_x = 20
        _start_y = 30
        self.write_text(f"Generation: {self._ga._generation}", _start_x, _start_y)
        self.write_text(f"Birds alive: {self._ga.num_alive}", _start_x, _start_y * 3)
        self.write_text(f"Score: {int(self._count / self._fps)}", _start_x, _start_y * 4)

    def add_ga(
        self,
        population_size: int,
        mutation_rate: float,
        x: int,
        y: int,
        size: int,
        hidden_layer_sizes: List[int],
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
            hidden_layer_sizes (List[int]): Neural network hidden layer sizes
            weights_range (List[float]): Range for random weights
            bias_range (List[float]): Range for random bias
        """
        self._ga = FlappyBirdGA.create(
            population_size, mutation_rate, x, y, size, hidden_layer_sizes, weights_range, bias_range
        )

    def update(self) -> None:
        """
        Run genetic algorithm, update Birds and draw to screen.
        """
        if self._count == self.max_count or self._ga.num_alive == 0:
            self._ga._analyse()
            self._ga._evolve()
            self._ga.reset()
            self._count = 0

        for _bird in self._ga._population._population:
            _bird.update()
            _bird.draw(self.screen)

        self._ga._evaluate()
        self._count += 1
        self._write_stats()
