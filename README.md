# 🚗 Deep Q-Network Navigation in PyBullet

This project builds upon the [fredsukkar/simple-car-env-template](https://github.com/fredsukkar/simple-car-env-template) and implements a **Deep Q-Network (DQN)** agent for autonomous navigation in a 2D **PyBullet** environment. The agent learns to reach dynamic goals while avoiding collisions, using **discrete actions**, **reward shaping**, and **visual diagnostics**.

> ✅ Extended and independently developed by Caesar Zhao (MSc Robotics, UTS)

---

## 🎯 Project Highlights

- **Extended Car Environment**: Built on top of `simple-car-env-template`, with enhancements in random obstacle generation and evaluation.
- **Deep Q-Network (DQN)**: Discrete policy with state-action value function approximation.
- **Reward Shaping**: Encourages reaching goals and penalizes collisions.
- **Evaluation Metrics**: Includes success rate, average reward, steps-to-goal.
- **Visual Diagnostics**: Evaluation animations, reward curves, and exploration insights.

---

## 🛠️ Environment Setup

```bash
conda create -n dqn_car python=3.10
conda activate dqn_car
pip install -r requirements.txt  # includes pybullet, gym, matplotlib, numpy, etc.
```

---

## ▶️ How to Run

1. **Train the Agent**

```bash
python train_dqn_agent.py
```

2. **Evaluate the Trained Policy**

```bash
python evaluate_agent.py
```

3. **Visualize Results**

- `evaluation_animation.gif`: Demo of a successful run
- `reward_curve.png`: Cumulative reward during training
- `epsilon_distribution.png`: Action exploration analysis

---

## 🧠 Methodology

- **Observation Space**: Local frame with relative goal and obstacles.
- **Action Space**: 9 discrete motion commands (linear & angular).
- **Reward Function**:
  - Positive reward on reaching goal
  - Penalties for collisions and prolonged paths
  - Shaped reward ∆d to encourage proximity to goal
- **Q-Network**:
  - MLP with 2 hidden layers of 128 units
  - Target network and experience replay
- **Exploration**: ε-greedy with decay

---

## 📊 Results (10 Evaluation Episodes)

| Metric | Value |
|--------|--------|
| **Success Rate** | 50% |
| **Avg Reward** | -38.23 |
| **Avg Steps to Goal** | 19.6 |

> Results indicate promising generalization under randomized obstacle setups.

---

## 📂 Repository Structure

```
RL_simple_car/
├── agent/
│   ├── dqnpy.py               # Core DQN agent logic
│   ├── evaluator.py           # Evaluation logic
│   ├── main.py                # Entry point to training
│   └── report_charts.py       # Plotting training results
├── simple_driving/
│   └── envs/simple_driving_env.py  # Modified PyBullet gym env
├── resources/                 # URDFs and geometry
├── data/                      # Saved figures and results
├── videos/                    # Evaluation animations
├── models/                    # Saved model checkpoints
├── dqn_navigation_pybullet.pdf  # Final project report
└── README.md                  # Project overview and usage
```

---

## 🙏 Acknowledgements

This project is based on the template from:
**fredsukkar/simple-car-env-template**  
🔗 https://github.com/fredsukkar/simple-car-env-template

---

## 🧑‍💻 Author

**Zhiye (Caesar) Zhao**  
MSc Robotics, University of Technology Sydney  
📧 [zhiye.zhao-1@student.uts.edu.au](mailto:zhiye.zhao-1@student.uts.edu.au)  
🌐 [Portfolio Website](https://caesar1457.github.io/zhiyezhao/)

---

## 📜 License

This project is provided for academic demonstration purposes only.  
© 2024 Caesar Zhao. All rights reserved.

