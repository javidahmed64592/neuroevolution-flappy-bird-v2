from __future__ import annotations

from flappy_bird.flappy_bird_app import FlappyBirdApp
from flappy_bird.pg.app import App

if __name__ == "__main__":
    fbga = FlappyBirdApp.create_app(
        name="Flappy Bird", width=1000, height=1000, fps=60, font="freesansbold.ttf", font_size=28, lifetime=200
    )
    fbga.add_ga(
        population_size=200,
        mutation_rate=4,
        x=30,
        y=500,
        size=20,
        nn_layer_sizes=[1, 2, 2],
        weights_range=[-1, 1],
        bias_range=[-1, 1],
    )
    fbga.run()
# TODO: Add Pipes
# TODO: Add App
