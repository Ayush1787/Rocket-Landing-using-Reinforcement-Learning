# 🚀 Rocket Landing using Reinforcement Learning (DQN)

## 📌 Project Overview

This project implements an intelligent rocket landing system using Deep Reinforcement Learning (DQN). The agent learns to control the rocket's movement such as thrust, rotation, and direction to achieve a safe landing.

The system uses a reward-based learning mechanism where the agent improves its performance over time by interacting with the environment.

---

## 🚀 Features

* 🧠 Reinforcement Learning based control system
* 🎮 Interactive simulation using Pygame
* 📊 Real-time learning and training
* ⚡ Deep Q-Network (DQN) implementation
* 🎯 Autonomous rocket landing
* 📉 Reward & loss visualization

---

## 🧠 Reinforcement Learning Concepts

### 1. Agent & Environment

* **Agent** → Rocket
* **Environment** → Simulation world

---

### 2. State

The state includes:

* Position (x, y)
* Velocity (vx, vy)
* Angle
* Angular velocity
* Fuel level

Defined in:
📄 

---

### 3. Actions

The agent can perform:

* Rotate left
* Rotate right
* Move left
* Move right

---

### 4. Reward Function

The reward system guides learning:

* ✅ Safe landing → High reward (up to +100)
* ❌ Crash → Negative reward (-20)
* 🔄 Excess rotation → Penalty
* ⛽ Fuel usage → Penalty

---

### 5. Q-Learning

The agent learns using the Q-value update rule:

Bellman Equation:

```
Q(s,a) = r + γ * max Q(s', a')
```

---

### 6. Deep Q-Network (DQN)

A neural network is used to approximate Q-values.

Implemented in:
📄 

---

### 7. Dueling DQN

The model separates:

* Value function
* Advantage function

This improves stability and performance.

---

### 8. Experience Replay

The agent stores past experiences and learns from them randomly.

Implemented in:
📄 

---

### 9. Target Network

A separate target network stabilizes training.

---

### 10. Exploration vs Exploitation

* Random actions (exploration)
* Best actions (exploitation)

Controlled by epsilon decay.

---

## ⚙️ Tech Stack

* Python 🐍
* PyTorch 🔥
* NumPy 📊
* Pygame 🎮
* Matplotlib 📈

---

## 📂 Project Structure

```id="2y1ymq"
rocket_rl/
│
├── Rocket.py
├── simulation.py
├── DQN.py
├── DQN_Agent.py
├── plot_results.py
├── rocket_model.pth
├── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash id="4lnbhi"
pip install pygame torch numpy matplotlib
```

---

### 2️⃣ Run Simulation

```bash id="iwsnhq"
python simulation.py
```

---

## 🎮 Simulation Details

The rocket operates in a constrained environment:

* Screen size: 800x600
* Gravity applied
* Wind disturbances included
* Fuel-based thrust system

Defined in:
📄 

---

## 📊 Training

The agent learns by:

1. Observing state
2. Taking action
3. Receiving reward
4. Updating Q-values

Training handled in:
📄 

---

## 📈 Visualization

Training performance can be visualized using:
📄 

* Episode rewards
* Loss curve

---

## 💡 Key Techniques Used

* Deep Q-Network (DQN)
* Dueling Architecture
* Experience Replay
* Target Network
* Reward Engineering
* Epsilon-Greedy Strategy

---

## 🎓 Viva Questions

### Q1: What is Reinforcement Learning?

👉 Learning through interaction using rewards

### Q2: What is DQN?

👉 Neural network-based Q-learning

### Q3: Why use Experience Replay?

👉 To break correlation and stabilize training

### Q4: What is Bellman Equation?

👉 Core equation for updating Q-values

---

## 🏁 Conclusion

This project demonstrates:

* Advanced AI control using RL
* Real-time simulation
* Autonomous decision-making

---

## 👨‍💻 Author

**Ayush Kumar**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
