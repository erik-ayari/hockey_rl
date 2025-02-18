import json
import argparse

from math import prod
from dataclasses import asdict

from utils import EnvironmentType, AgentType, SplitActionSpace
from hockey.hockey_env import HockeyEnv, HockeyEnv_BasicOpponent
from .lightning import SoftActorCritic

import gymnasium as gym
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.callbacks import ModelCheckpoint

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_environment(env_config: dict):
    # Retrieves the specified environment, uses hockey if none is specified
    env_type = env_config.get('type', 'hockey')
    # Additional keyword arguments (e.g. whether to use weak opponent)
    kwargs = env_config.get('kwargs', {})

    if env_type == 'Hockey':
        split = SplitActionSpace.SPLIT
        return HockeyEnv(**kwargs), EnvironmentType.GAME, AgentType.MULTI_AGENT, split
    elif env_type == 'Hockey_BasicOpponent':
        return HockeyEnv_BasicOpponent(**kwargs), EnvironmentType.GAME, AgentType.SINGLE_AGENT, SplitActionSpace.NO_SPLIT
    else:
        try:
            return gym.make(env_type, **kwargs), EnvironmentType.REGULAR, AgentType.SINGLE_AGENT, SplitActionSpace.NO_SPLIT
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
    env, environment_type, agent_type, split_action_space = create_environment(env_config)
    
    # Initialize model with hyperparameters from config
    model = SoftActorCritic(
        env,
        environment_type,
        agent_type,
        split_action_space,
        config["model"]
    )

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir=config["logger"]["save_dir"],
        name=config["logger"]["name"]
    )

    # Create a ModelCheckpoint callback that saves a checkpoint after every validation
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config['logger']['save_dir']}/{config['logger']['name']}",
        filename="{epoch}-{step}",
        save_top_k=-1,  # Save all checkpoints, do not delete previous ones
    )

    trainer = pl.Trainer(
        accelerator             = config["training"]["accelerator"],
        log_every_n_steps       = config["training"]["log_every_n_steps"],
        check_val_every_n_epoch = config["training"]["check_val_every_n_epoch"],
        max_epochs              = config["training"]["max_epochs"],
        logger                  = logger,
        callbacks               = [checkpoint_callback]
    )

    trainer.fit(model)

if __name__ == "__main__":
    main()