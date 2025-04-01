import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Create output directories (e.g., data/graphs and data/videos)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data")
graph_dir = os.path.join(data_dir, "graphs")
video_dir = os.path.join(data_dir, "videos")
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)


def plot_training_curve(reward_history, smooth_window=10, save_path="training_reward_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label="Raw Reward", alpha=0.4)

    if len(reward_history) >= smooth_window:
        smooth = np.convolve(reward_history, np.ones(smooth_window)/smooth_window, mode="valid")
        plt.plot(np.arange(smooth_window - 1, len(reward_history)), smooth,
                 label=f"Smoothed (window={smooth_window})", color="orange")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✓] Training reward curve saved to: {save_path}")
    plt.close()


def create_evaluation_gif(frames, save_path="evaluation_animation.gif", fps=10):
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"[✓] Evaluation animation saved as: {save_path}")


def plot_evaluation_metrics(evaluation_result, save_path="evaluation_metrics.png"):
    metrics = {
        "Avg Reward": evaluation_result.get("average_reward", 0),
        "Success Rate (%)": evaluation_result.get("success_rate", 0) * 100,
        "Avg Steps": evaluation_result.get("avg_steps_to_success", 0)
    }
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=["blue", "green", "orange"])
    plt.title("Evaluation Metrics")
    plt.ylabel("Value")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05 * yval, f"{yval:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✓] Evaluation metrics saved to: {save_path}")
    plt.close()


def plot_epsilon_distribution(probabilities, save_path="epsilon_distribution.png"):
    labels = [f"Action {i}" for i in range(len(probabilities))]

    plt.figure(figsize=(8, 8))
    plt.pie(probabilities, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Epsilon-Greedy Exploration Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✓] Epsilon distribution saved to: {save_path}")
    plt.close()

