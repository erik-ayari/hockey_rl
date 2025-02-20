import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def extract_rewards_from_event_file(file_path, tag='val_cumulative_reward'):
    """
    Extract steps and scalar values for a given tag from a TensorBoard event file.
    """
    ea = event_accumulator.EventAccumulator(file_path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        print(f"Tag '{tag}' not found in {file_path}")
        return None, None
    events = ea.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    return steps, values

def aggregate_runs(global_folder, tag='val_cumulative_reward'):
    """
    Aggregate the scalar data across all version folders within a global folder.
    """
    # Find all version directories (assumed to start with 'version_')
    version_dirs = [os.path.join(global_folder, d) for d in os.listdir(global_folder) if d.startswith('version_')]
    
    all_runs_steps = []
    all_runs_values = []
    
    for version_dir in version_dirs:
        # Locate the event file in the version folder (ignoring the checkpoints folder)
        event_files = glob.glob(os.path.join(version_dir, 'events.out.tfevents.*'))
        if not event_files:
            print(f"No event file found in {version_dir}")
            continue
        event_file = event_files[0]  # assume one event file per version directory
        steps, values = extract_rewards_from_event_file(event_file, tag)
        if steps is None or values is None:
            continue
        all_runs_steps.append(steps)
        all_runs_values.append(values)
    
    if not all_runs_steps:
        print(f"No runs with tag '{tag}' found in {global_folder}")
        return None, None, None

    # Assume that all runs logged at the same steps; if not, use the intersection of steps.
    common_steps = set(all_runs_steps[0])
    for steps in all_runs_steps[1:]:
        common_steps = common_steps.intersection(steps)
    common_steps = sorted(common_steps)
    
    if not common_steps:
        print("No common steps found across runs.")
        return None, None, None

    # Align values to the common steps
    run_values = []
    for steps, values in zip(all_runs_steps, all_runs_values):
        step_to_value = dict(zip(steps, values))
        run_values.append([step_to_value[step] for step in common_steps])
    
    run_values = np.array(run_values)
    mean_values = np.mean(run_values, axis=0)
    std_values = np.std(run_values, axis=0)
    
    return common_steps, mean_values, std_values

def plot_performance(steps, mean_values, std_values, env_name, tag='val_cumulative_reward', save_dir=None):
    """
    Create a ggplot-styled plot showing the average performance with standard deviation shading.
    """
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mean_values, label='Average Cumulative Reward', lw=2)
    plt.fill_between(steps, 
                     np.array(mean_values) - np.array(std_values),
                     np.array(mean_values) + np.array(std_values),
                     alpha=0.3, label='Std Dev')
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.title(f"{env_name}")
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{env_name}_performance.pdf")
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot average performance with standard deviation from TensorBoard logs."
    )
    parser.add_argument(
        "global_folders", nargs='+', 
        help="Paths to global folders (each corresponding to one environment)."
    )
    parser.add_argument(
        "--tag", type=str, default="val_cumulative_reward",
        help="TensorBoard tag to extract (default: 'val_cumulative_reward')."
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Directory to save the plots. If not provided, plots are shown interactively."
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Maximum step value to plot. Data with steps greater than this will be ignored."
    )
    parser.add_argument(
        "--env_names", nargs='+'
    )
    args = parser.parse_args()

    for i, global_folder in enumerate(args.global_folders):
        env_name = args.env_names[i]
        print(f"Processing environment: {env_name}")
        steps, mean_values, std_values = aggregate_runs(global_folder, args.tag)
        if steps is None:
            continue

        # Filter the data based on max_steps if provided
        if args.max_steps is not None:
            steps_array = np.array(steps)
            mask = steps_array <= args.max_steps
            if not np.any(mask):
                print(f"No data points with steps <= {args.max_steps} for {env_name}. Skipping.")
                continue
            steps = steps_array[mask]
            mean_values = mean_values[mask]
            std_values = std_values[mask]

        plot_performance(steps, mean_values, std_values, env_name, tag=args.tag, save_dir=args.save)

if __name__ == "__main__":
    main()