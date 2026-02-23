"""Pygame application base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pygame
from pygame.locals import QUIT


class App(ABC):
    """This class can be used to create a Pygame application.

    Override the `update()` method for your application.
    """

    def __init__(self, name: str, width: int, height: int, fps: int, font: str, font_size: int) -> None:
        """Initialise App and set parameters.

        :param str name: App name
        :param int width: Screen width
        :param int height: Screen height
        :param int fps: Game FPS
        :param str font: Font style
        :param int font_size: Font size
        """
        self._name = name
        self._width = width
        self._height = height
        self._fps = fps
        self._font = font
        self._font_size = font_size
        self._running = False

    @classmethod
    def create_app(cls, name: str, width: int, height: int, fps: int, font: str, font_size: int) -> App:
        """Create application using app config.

        :param str name: App name
        :param int width: Screen width
        :param int height: Screen height
        :param int fps: Game FPS
        :param str font: Font style
        :param int font_size: Font size
        :return App: Pygame application
        """
        pygame.init()
        app = cls(name, width, height, fps, font, font_size)
        app._configure()
        return app

    @property
    def screen(self) -> pygame.Surface:
        """Get the Pygame display surface."""
        return self._display_surf

    def _configure(self) -> None:
        """Configure Pygame application."""
        pygame.display.set_caption(self._name)
        self._display_surf = pygame.display.set_mode((self._width, self._height))
        self._pg_font = pygame.font.SysFont(self._font, self._font_size)
        self._clock = pygame.time.Clock()

    def write_text(self, text: str, x: float, y: float) -> None:
        """Write text to the screen at the given position.

        :param str text: Text to write
        :param float x: x coordinate of text's position
        :param float y: y coordinate of text's position
        """
        _text = self._pg_font.render(text, 1, (255, 255, 255))
        self._display_surf.blit(_text, (x, y))

    @abstractmethod
    def update(self) -> None:
        """Update the game state."""
        pass

    def run(self) -> None:
        """Run the application and handle events."""
        self._running = True
        while self._running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    self._running = False
                    return

            self._display_surf.fill((0, 0, 0))

            self.update()
            pygame.display.update()
            self._clock.tick(self._fps)
