from __future__ import annotations

from flappy_bird.flappy_bird_app import FlappyBirdApp

if __name__ == "__main__":
    fbga = FlappyBirdApp.create_game(
        name="Flappy Bird", width=600, height=600, fps=60, font="freesansbold.ttf", font_size=24, lifetime=10
    )
    fbga.add_ga(
        population_size=100,
        mutation_rate=0.02,
        x=30,
        y=200,
        size=20,
        nn_layer_sizes=[1, 1, 2],
        weights_range=[-1, 1],
        bias_range=[-1, 1],
    )
    fbga.run()
