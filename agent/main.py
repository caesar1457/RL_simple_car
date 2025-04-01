# src/RL_simple_car/agent/main.py

import os
import sys
import torch
import gym
import time

# === 0. Add simple_driving to sys.path to support environment registration ===
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# === 0.1 Register custom Gym environment ===
from gym.envs.registration import register
try:
    register(
        id="SimpleDriving-v0",
        entry_point="simple_driving.envs.simple_driving_env:SimpleDrivingEnv"
    )
except gym.error.Error:
    pass  # Already registered

# === 1. Import modules ===
from dqn import QNetwork, train_q_learning
from evaluator import Evaluator
from report_charts import (
    plot_training_curve,
    plot_evaluation_metrics,
    plot_epsilon_distribution,
    create_evaluation_gif
)
import imageio

OUTPUT_DIM = 9
HIDDEN_DIM = 128
NUM_OBSTACLES = 3
INPUT_DIM = 2 + 2 * NUM_OBSTACLES

# === Config ===
show_plot = False  # Toggle this to control whether to display final figures interactively

if __name__ == "__main__":
    # === 2. Create training environment ===
    print("🖥️ Creating training environment...")
    train_env = gym.make("SimpleDriving-v0", 
                         apply_api_compatibility=True, 
                         renders=False, 
                         isDiscrete=True, 
                         num_obstacles=NUM_OBSTACLES
                         )

    # === 3. Set up directories and timestamp ===
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    data_dir = os.path.join(current_dir, "..", "data")
    model_dir = os.path.join(data_dir, "models")
    video_dir = os.path.join(data_dir, "videos")
    graph_dir = os.path.join(data_dir, "graphs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"simple_driving_qlearning_{timestamp}.pth")
    curve_path = os.path.join(graph_dir, f"training_curve_{timestamp}.png")
    gif_path = os.path.join(graph_dir, f"evaluation_animation_{timestamp}.gif")
    metrics_path = os.path.join(graph_dir, f"evaluation_metrics_{timestamp}.png")
    epsilon_path = os.path.join(graph_dir, f"epsilon_distribution_{timestamp}.png")

    # === 4. Train model ===
    print("🚗 Starting training of DQN model...")
    trained_model, reward_history = train_q_learning(
                                                        train_env,
                                                        input_dim=INPUT_DIM,
                                                        output_dim=OUTPUT_DIM,
                                                        hidden_dim=HIDDEN_DIM,
                                                        episodes=500,
                                                        max_steps=200
                                                    )
    torch.save(trained_model.state_dict(), model_path)
    print(f"✅ Model saved to: {os.path.relpath(model_path)}")
    train_env.close()

    # === 5. Reload model for evaluation ===
    print("\n📦 Loading model for evaluation...")
    loaded_model = QNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    print("✅ Model loaded successfully.\n")

    # === 6. Create evaluation environment for video ===
    print("🎥 Creating evaluation environment (third-person view)...")
    eval_env = gym.make("SimpleDriving-v0", 
                        apply_api_compatibility=True, 
                        renders=False, 
                        isDiscrete=True, 
                        render_mode='tp_camera',
                        num_obstacles=NUM_OBSTACLES
                        )

    # === 7. Run and record ONE episode ===
    print("🎬 Running evaluation episode and capturing frames...")
    state, _ = eval_env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    frames = []
    total_reward = 0

    for step in range(200):
        with torch.no_grad():
            action = torch.argmax(loaded_model(state)).item()
        next_state, reward, done, _, _ = eval_env.step(action)
        state = torch.tensor(next_state, dtype=torch.float32)
        total_reward += reward
        frame = eval_env.render()
        if frame is not None:
            frames.append(frame)
        if done:
            break

    eval_env.close()
    print(f"🎯 Total evaluation reward: {total_reward:.2f}")

    # === 8. Create new env for evaluation metrics ===
    metrics_env = gym.make("SimpleDriving-v0", 
                           apply_api_compatibility=True, 
                           renders=False, 
                           isDiscrete=True,
                           num_obstacles=NUM_OBSTACLES
                           )
    evaluator = Evaluator(metrics_env, loaded_model, success_threshold=100.0)
    evaluation_result = evaluator.evaluate(episodes=10, max_steps=200, epsilon=0.0, plot=False)
    metrics_env.close()

    # === 9. Save training and evaluation charts ===
    plot_training_curve(reward_history, save_path=curve_path)
    plot_evaluation_metrics(evaluation_result, save_path=metrics_path)
    create_evaluation_gif(frames, save_path=gif_path)
    mp4_path = os.path.join(video_dir, f"evaluation_episode_{timestamp}.mp4")
    imageio.mimsave(mp4_path, frames, fps=30)
    print(f"[✓] MP4 video also saved to: {mp4_path}")

    # Optional: ε-greedy distribution if used non-uniform
    action_probs = [0.25, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05, 0.025, 0.025]
    plot_epsilon_distribution(action_probs, save_path=epsilon_path)

    # === 10. Optionally show plots interactively ===
    if show_plot:
        import matplotlib.pyplot as plt
        img = plt.imread(curve_path)
        plt.imshow(img)
        plt.title("Training Reward Curve")
        plt.axis("off")
        plt.show()
