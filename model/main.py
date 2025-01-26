import json
import argparse

from math import prod
from dataclasses import asdict

from hockey.hockey_env import HockeyEnv_BasicOpponent
from .lightning import SoftActorCritic

import gymnasium as gym
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_environment(env_config: dict):
    # Retrieves the specified environment, uses hockey if none is specified
    env_type = env_config.get('type', 'hockey')
    # Additional keyword arguments (e.g. whether to use weak opponent)
    kwargs = env_config.get('kwargs', {})

    # Returns additional bool to indicate whether validation is win rate or reward
    if env_type.lower() == 'hockey':
        return HockeyEnv_BasicOpponent(**kwargs), True
    else:
        try:
            return gym.make(env_type, **kwargs), False
        except gym.error.Error as e:
            raise ValueError(f"Failed to create environment '{env_type}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Soft Actor-Critic Training")
    parser.add_argument(
        '--config',
        '-c',
        type=str,
        help='Path to the JSON configuration file. (Usually "configs/...json")'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize environment, and validation type
    env_config = config.get('environment', {})
    env, game_validation = create_environment(env_config)
    config["model"]["game_validation"] = game_validation
    
    # Initialize model with hyperparameters from config
    model = SoftActorCritic(env, config["model"])

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir=config["logger"]["save_dir"],
        name=config["logger"]["name"]
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        log_every_n_steps=1,
        check_val_every_n_epoch=1_000,
        max_epochs=500_000,
        logger=logger
    )

    trainer = pl.Trainer(
        accelerator             = config["training"]["accelerator"],
        log_every_n_steps       = config["training"]["log_every_n_steps"],
        check_val_every_n_epoch = config["training"]["check_val_every_n_epoch"],
        max_epochs              = config["training"]["max_epochs"],
        logger                  = logger
    )

    trainer.fit(model)

if __name__ == "__main__":
    main()