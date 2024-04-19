from __future__ import annotations

import json

from flappy_bird.flappy_bird_app import FlappyBirdApp

CONFIG_FILEPATH = "./config/config.json"


def load_config(filepath: str) -> dict:
    """
    Load config from `json` file.

    Parameters:
        filepath (str): Path to config file

    Returns:
        data (dict): Application config
    """
    with open(filepath) as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    config = load_config(CONFIG_FILEPATH)
    app_config = config["app"]
    ga_config = config["genetic_algorithm"]

    fba = FlappyBirdApp.create_game(
        name=app_config["name"],
        width=app_config["width"],
        height=app_config["height"],
        fps=app_config["fps"],
        font=app_config["font"],
        font_size=app_config["font_size"],
        lifetime=app_config["lifetime"],
    )
    fba.add_ga(
        population_size=ga_config["population_size"],
        mutation_rate=ga_config["mutation_rate"],
        bird_x=ga_config["bird_x"],
        bird_y=ga_config["bird_y"],
        bird_size=ga_config["bird_size"],
        hidden_layer_sizes=ga_config["hidden_layer_sizes"],
        weights_range=ga_config["weights_range"],
        bias_range=ga_config["bias_range"],
    )
    fba.run()
