import argparse
import numpy as np
import gymnasium as gym

from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


def make_env(weak_opponent=True, verbose=False):
    def _init():
        env = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=weak_opponent)
        env.verbose = verbose
        return env
    return _init

def train_sac_agent(weak_opponent):
    """
    Train a SAC agent against the built-in BasicOpponent in the hockey environment.
    """

    # Create a vectorized environment using DummyVecEnv 
    # (Stable-Baselines3 requires a VecEnv for most algorithms).
    env = DummyVecEnv([make_env(weak_opponent=weak_opponent)])  # optionally use weak_opponent=False

    # Initialize the SAC model
    model = SAC(
        policy=MlpPolicy,
        env=env,
        verbose=1,
        # Further experiments:
        # learning_rate=3e-4,
        # buffer_size=100000,
        # batch_size=256,
        # tau=0.02,
        # ent_coef="auto",
        # gamma=0.99,
        # etc.
    )

    print(model.policy)
    exit()

    # Train the agent
    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save("sac_hockey_model")

    # Evaluate the trained policy
    eval_env = DummyVecEnv([make_env(weak_opponent=True)])
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"Evaluation reward over 20 episodes: mean={mean_reward:.2f} +/- {std_reward:.2f}")

    # obs, _info = eval_env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, truncs, info = eval_env.step(action)
    #     eval_env.render()
    #     if any(dones) or any(truncs):
    #         obs, _info = eval_env.reset()

    env.close()
    eval_env.close()

def main():
    parser = argparse.ArgumentParser(description="Train an SAC agent on HockeyEnv.")
    parser.add_argument("--opponent", type=str, default="weak",
                        choices=["weak", "strong"],
                        help="Opponent type: 'weak' or 'strong'.")
    args = parser.parse_args()
    if args.opponent == "weak":
        weak_opponent = True
    elif args.opponent == "strong":
        weak_opponent = False
    else:
        raise ValueError("Invalid Opponent")
    train_sac_agent(weak_opponent=weak_opponent)

if __name__ == "__main__":
    main()