from __future__ import annotations

import pygame
from pygame.locals import QUIT


class App:
    """
    This class can be used to create a Pygame application.

    Override the `update()` method and optionally the `run()` method to create a specific app.
    """

    def __init__(self, name: str, width: int, height: int, fps: int, font: str, font_size: int) -> None:
        """
        Initialise App and set parameters.

        Parameters:
            name (str): App name
            width (int): Screen width
            height (int): Screen height
            fps (int): Game FPS
            font (str): Font style
            font_size (int): Font size
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
        """
        Create application using app config.

        Parameters:
            name (str): App name
            width (int): Screen width
            height (int): Screen height
            fps (int): Game FPS
            font (str): Font style
            font_size (int): Font size

        Returns:
            _app (App): App with screen, clock, and font set.
        """
        pygame.init()
        app = cls(name=name, width=width, height=height, fps=fps, font=font, font_size=font_size)
        app._configure()
        return app

    def _configure(self) -> None:
        """
        Configure Pygame application.
        """
        pygame.display.set_caption(self._name)
        self._display_surf = pygame.display.set_mode((self._width, self._height))
        self._pg_font = pygame.font.SysFont(self._font, self._font_size)
        self._clock = pygame.time.Clock()

    def write_text(self, text: str, x: float, y: float) -> None:
        """
        Write text to the screen at the given position.

        Parameters:
            text (str): Text to write
            x (float): x coordinate of text's position
            y (float): y coordinate of text's position
        """
        _text = self._pg_font.render(text, False, (255, 255, 255))
        self._display_surf.blit(_text, (x, y))

    def update(self) -> None:
        """
        Display application information to screen.
        """
        start_x = 50
        start_y = 50
        self.write_text("App info:", start_x, start_y)
        self.write_text(f"Name: {self._name}", start_x, start_y * 3)
        self.write_text(f"Width: {self._width}", start_x, start_y * 4)
        self.write_text(f"Height: {self._height}", start_x, start_y * 5)
        self.write_text(f"Font: {self._font}", start_x, start_y * 6)
        self.write_text(f"Font size: {self._font_size}", start_x, start_y * 7)
        self.write_text(f"FPS: {self._clock.get_fps()}", start_x, start_y * 9)

    def run(self) -> None:
        """
        Run the application and handle events.
        """
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
