import argparse
import gymnasium as gym
import numpy as np

from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode
from stable_baselines3 import SAC

def make_env(weak_opponent=True):
    """
    Return a HockeyEnv_BasicOpponent environment with either weak or strong opponent.
    """
    env = HockeyEnv_BasicOpponent(
        mode=Mode.NORMAL,  # Normal game mode
        weak_opponent=weak_opponent
    )
    return env

def run_validation(
    model_path, 
    n_episodes=10, 
    opponent_strength="weak", 
    render=False
    ):
    # 1) Decide on opponent
    if opponent_strength.lower() == "weak":
        env = make_env(weak_opponent=True)
    elif opponent_strength.lower() == "strong":
        env = make_env(weak_opponent=False)
    else:
        raise ValueError("opponent_strength must be either 'weak' or 'strong'.")

    # 2) Load the trained SAC model
    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path)

    # 3) Run evaluation episodes
    total_rewards = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward

            if render:
                env.render()

        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: reward = {episode_reward:.2f}")

    env.close()

    # 4) Print summary statistics
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nValidation over {n_episodes} episodes")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Validate a trained SAC agent on HockeyEnv.")
    parser.add_argument("--model-path", type=str, default="sac_hockey_model.zip",
                        help="Path to the saved model (zip file).")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run.")
    parser.add_argument("--opponent", type=str, default="weak",
                        choices=["weak", "strong"],
                        help="Opponent type: 'weak' or 'strong'.")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment if set.")

    args = parser.parse_args()
    run_validation(
        model_path=args.model_path,
        n_episodes=args.episodes,
        opponent_strength=args.opponent,
        render=args.render
    )

if __name__ == "__main__":
    main()