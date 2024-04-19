from typing import ClassVar

import numpy as np
import pygame


class Pipe:
    """
    Pipe to use in Flappy Bird app.
    """

    WIDTH = 50
    SPACING = 220
    START_SPEED = 3.5
    MAX_SPEED = 11
    ACC_SPEED = 0.03
    START_SPAWNTIME = 120
    MIN_SPAWNTIME = 80
    ACC_SPAWNTIME = 1
    MIN_HEIGHT = 20
    COLOUR: ClassVar = [0, 200, 0]
    X_LIM: float
    Y_LIM: float

    def __init__(self, speed: float) -> None:
        """
        Initialise Pipe with speed to move across the screen.

        Parameters:
            speed (float): Pipe movement speed
        """
        self._x = self.X_LIM
        self._top_height = np.random.uniform(low=self.MIN_HEIGHT, high=(self.Y_LIM - self.MIN_HEIGHT - self.SPACING))
        self._bottom_height = self.Y_LIM - self._top_height + self.SPACING
        self._speed = speed

    @property
    def rects(self) -> list[pygame.Rect]:
        _top_pipe = pygame.Rect(self.top_pos[0], self.top_pos[1], self.WIDTH, self._top_height)
        _bottom_pipe = pygame.Rect(self.bottom_pos[0], self.bottom_pos[1], self.WIDTH, self._bottom_height)
        return [_top_pipe, _bottom_pipe]

    @property
    def top_pos(self) -> list[float]:
        return [self._x, 0]

    @property
    def bottom_pos(self) -> list[float]:
        return [self._x, self._top_height + self.SPACING]

    @property
    def offscreen(self) -> bool:
        return self._x < -self.WIDTH

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw Pipe on the display.

        Parameters:
            screen (Surface): Screen to draw Pipe to
        """
        if self.offscreen:
            return
        pygame.draw.rect(screen, self.COLOUR, self.rects[0])
        pygame.draw.rect(screen, self.COLOUR, self.rects[1])

    def update(self) -> None:
        """
        Move Pipe.
        """
        if self.offscreen:
            return
        self._x -= self._speed

    @staticmethod
    def get_speed(pipes_spawned: int) -> float:
        """
        Get speed for Pipe based on how many have spawned in current generation.

        Parameters:
            pipes_spawned (int): Number of Pipes spawned in current generation

        Returns:
            speed (float): Pipe speed
        """
        speed = min(Pipe.START_SPEED + (pipes_spawned * Pipe.ACC_SPEED), Pipe.MAX_SPEED)
        return speed

    @staticmethod
    def get_spawn_time(pipes_spawned: int) -> float:
        """
        Get time for Pipe to spawn based on how many have spawned in current generation.

        Parameters:
            pipes_spawned (int): Number of Pipes spawned in current generation

        Returns:
            spawn_time (float): Time until Pipe spawns
        """
        spawn_time = max(Pipe.START_SPAWNTIME - (pipes_spawned * Pipe.ACC_SPAWNTIME), Pipe.MIN_SPAWNTIME)
        return spawn_time
