import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
import os

# ===============================
# Part 1: QNetwork
# ===============================
class QNetwork(nn.Module):
    """
    A simple multilayer perceptron to estimate Q-values for each action.
    Input is a 2D state (relative position to the goal), output is Q-values for 9 discrete actions.
    input_dim: Dimension of input state (2)
    output_dim: Dimension of output Q-values (9)
    hidden_dim: Dimension of hidden 128
    """
    def __init__(self, input_dim=8, output_dim=9, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input layer -> Hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Hidden layer -> Output layer
        )

    def forward(self, x):
        return self.model(x)

# ===============================
# Part 2: ε-greedy Action Selection
# ===============================
def select_action(model, state, epsilon, env):
    """
    Select an action based on the ε-greedy strategy:
      - With probability epsilon, choose a random action (exploration)
      - Otherwise, choose the action with the highest Q-value (exploitation)
    Parameters:
        model: QNetwork model
        state: Current state (must be torch.tensor)
        epsilon: Exploration rate
        env: Gym environment (used for sampling random actions)
    Returns:
        action: Chosen action index (integer)
    """
    if np.random.rand() < epsilon:
        # Prefer forward-moving actions more than others
        action_probs = np.array([0.25, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05, 0.025, 0.025])
        action_probs /= action_probs.sum()  # Normalize
        action = np.random.choice(np.arange(env.action_space.n), p=action_probs)
    else:
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()
    return action

# ===============================
# Part 3: Q-learning Training Loop
# ===============================
def train_q_learning(env, input_dim, output_dim, hidden_dim, episodes=500, max_steps=200):
    """
    Train the agent using Q-learning. During training, MSELoss is used to calculate TD error,
    and the Adam optimizer is used to update network parameters.
    At each time step, Q-values are updated using the Bellman equation:
        target = reward + gamma * max(Q(next_state)) (if not done)
    The ε-greedy strategy is used for exploration.
    
    Returns:
        model: Trained QNetwork model
        reward_history: Total reward per episode (list)
    """
    model = QNetwork(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    gamma = 0.99

    reward_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0

        for step in range(max_steps):
            action = select_action(model, state, epsilon, env)
            next_state, reward, done, _, _ = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            with torch.no_grad():
                q_next = model(next_state_tensor).max().item()
            target = reward + gamma * q_next * (1 - int(done))

            pred = model(state)[action]
            loss = criterion(pred, torch.tensor(target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state_tensor
            total_reward += reward

            if done:
                break

        reward_history.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % 50 == 0:
            print(f"Episode {ep}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return model, reward_history

# ===============================
# Part 4: Training Reward Plotting
# ===============================
def plot_training_curve(reward_history, save_path=None, smooth_window=10):
    """
    Plot the total reward curve during training, with optional moving average smoothing.

    Parameters:
        reward_history: List of total rewards per episode
        save_path: If specified, saves the plot to this path
        smooth_window: Window size for moving average smoothing
    """
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label="Raw Reward", alpha=0.4)

    if len(reward_history) >= smooth_window:
        smooth = np.convolve(reward_history, np.ones(smooth_window)/smooth_window, mode="valid")
        plt.plot(np.arange(smooth_window - 1, len(reward_history)), smooth,
                 label=f"Smoothed (window={smooth_window})", color='orange')

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📊 Training reward curve saved to: {save_path}")
    else:
        plt.show()
