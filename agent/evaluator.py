import torch
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# Part 1: Evaluation (Model Testing)
# ===============================
class Evaluator:
    def __init__(self, env, model, success_threshold=100.0):
        """
        Parameters:
            env: Gym environment object
            model: Trained QNetwork model
            success_threshold: Reward threshold to determine if the task is successful
        """
        self.env = env
        self.model = model
        self.success_threshold = success_threshold

    def evaluate(self, episodes=10, max_steps=200, epsilon=0.0, plot=True):
        """
        Evaluate the model's performance over multiple episodes.
        Returns evaluation metrics.
        """
        rewards = []
        steps_to_success = []
        success_count = 0

        for ep in range(episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            total_reward = 0
            step_count = 0

            for step in range(max_steps):
                action = self.select_action(state, epsilon)
                next_state, reward, done, _, _ = self.env.step(action)
                state = torch.tensor(next_state, dtype=torch.float32)
                total_reward += reward
                step_count += 1
                if done:
                    break

            rewards.append(total_reward)
            if total_reward >= self.success_threshold:
                success_count += 1
                steps_to_success.append(step_count)

            print(f"[Episode {ep+1}] Total Reward: {total_reward:.2f}, Steps: {step_count}")

        avg_reward = np.mean(rewards)
        avg_steps = np.mean(steps_to_success) if steps_to_success else float('nan')
        success_rate = success_count / episodes

        print("\n📊 Evaluation Summary:")
        print(f"✅ Average Reward      : {avg_reward:.2f}")
        print(f"✅ Success Rate        : {success_rate * 100:.1f}%")
        print(f"✅ Avg Steps to Success: {avg_steps:.1f}")

        if plot:
            self._plot_rewards(rewards)

        return {
            'average_reward': avg_reward,
            'success_rate': success_rate,
            'avg_steps_to_success': avg_steps,
            'rewards': rewards
        }

    def select_action(self, state, epsilon):
        """
        Select action using ε-greedy strategy
        """
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.model(state)
                return torch.argmax(q_values).item()

    def _plot_rewards(self, rewards):
        plt.figure(figsize=(8, 4))
        plt.plot(rewards, marker='o')
        plt.title("Episode Reward Curve")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid()
        plt.tight_layout()
        plt.show()
