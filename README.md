# Self-Improving Queue Management System

## 📖 Project Overview
This project is developed for the **Self-Improving AI** course. It implements a **Reinforcement Learning (Q-Learning)** agent to optimize a service queue system.

The core challenge in Service Operations is the trade-off between:
1.  **Minimizing Wait Time:** Customers dislike waiting (requires fast/expensive servers).
2.  **Minimizing Operational Cost:** High-speed servers consume more energy and resources.

Unlike traditional rule-based systems (e.g., "if queue > 5 then switch"), this agent starts with **zero knowledge** and learns the optimal control policy through trial-and-error, demonstrating the capability of **Self-Improving Systems**.

---

## ⚙️ Mathematical Modeling (MDP)

The problem is formulated as a **Markov Decision Process (MDP)** tuple $\langle S, A, P, R \rangle$:

### 1. State Space ($S$)
The state represents the current congestion level:
* $S = \{0, 1, 2, ..., \text{MAX\_QUEUE}\}$
* (Default `MAX_QUEUE = 15`)

### 2. Action Space ($A$)
The agent controls the service rate at each time step:
* **Action 0 (Slow Mode):** Low energy cost, low service rate ($\mu=0.2$).
* **Action 1 (Fast Mode):** High energy cost, high service rate ($\mu=0.6$).

### 3. Reward Function ($R$)
To enable self-improvement, the Reward Function penalizes both waiting customers and energy usage. Since RL maximizes rewards, we define it as negative cost:

$$R_t = - ( \text{WaitCost} + \text{OperationalCost} )$$
$$R_t = - ( 1.0 \times \text{QueueLength}_t + 2.0 \times \mathbb{I}(\text{Action}=\text{Fast}) )$$

* **Wait Cost:** 1.0 per person per minute.
* **Energy Cost:** 2.0 per minute when Fast Mode is active.

---

## 📂 Project Structure

```text
.
├── main.py              # Core implementation (Environment + Q-Learning Agent)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── learning_curve.png   # (Generated) Visualization of the learning process

## 🚀 How to Run

Prerequisites: Python 3.x.  
First, install the required dependencies by running `pip install -r requirements.txt` in your terminal.  
Then run the simulation by executing `python main.py`. This will train the Q-learning agent and automatically generate the learning curve image (`learning_curve.png`).

## 📊 Results & Interpretation

After training, the system produces two main outputs.  
The first output is the learning curve (`learning_curve.png`), which shows the total reward per episode. In the early episodes, rewards are highly unstable and relatively low because the agent is exploring and making suboptimal decisions. In later episodes, the curve stabilizes at a higher reward level, indicating that the agent has learned an effective policy.  
The second output is the learned policy derived from the final Q-table, which is printed in the console. A typical converged policy is that when the queue length is between 0 and 1, the agent selects Slow Mode to save energy, and when the queue length is 2 or higher, the agent selects Fast Mode to reduce waiting costs. This switching behavior is learned automatically without any predefined rules.

## 👤 Author

Name: Jiarong Chen  
Course: Self-Improving AI  
Topic: Queue Management for Service Systems using MDP