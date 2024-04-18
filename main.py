from __future__ import annotations

from flappy_bird.flappy_bird_app import FlappyBirdApp

if __name__ == "__main__":
    fba = FlappyBirdApp.create_game(
        name="Flappy Bird", width=500, height=800, fps=60, font="freesansbold.ttf", font_size=24, lifetime=10
    )
    fba.add_ga(
        population_size=200,
        mutation_rate=0.02,
        x=40,
        y=250,
        size=40,
        hidden_layer_sizes=[1],
        weights_range=[-1, 1],
        bias_range=[-1, 1],
    )
    fba.run()
