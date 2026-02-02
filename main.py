import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# Part 1: Environment Definition (The World)
# ==========================================
class ServiceQueueEnv:
    """
    Simulates a simple single-server queue system.
    
    State Space: 
        Current number of people in the queue (0 to MAX_QUEUE).
        
    Action Space: 
        0 = Slow Mode (Low operational cost, slow service rate).
        1 = Fast Mode (High operational cost, fast service rate).
    """
    def __init__(self, max_queue=10):
        self.max_queue = max_queue
        self.state = 0  # Initial queue length
        
        # Simulation Parameters
        self.arrival_prob = 0.3      # 30% probability of a new arrival per time step
        self.service_rate_slow = 0.2 # Slow Mode: 20% chance to serve a customer
        self.service_rate_fast = 0.6 # Fast Mode: 60% chance to serve a customer
        
        # Cost Parameters (Used for Reward Calculation)
        self.cost_per_person = 1.0   # Penalty for each person waiting in line
        self.cost_energy_fast = 2.0  # Extra energy penalty for running Fast Mode

    def reset(self):
        """Resets the environment to the initial state."""
        self.state = 0
        return self.state

    def step(self, action):
        """
        Proceeds the simulation by one time step.
        Returns: next_state, reward
        """
        # 1. Simulate Arrival (Bernoulli process)
        if random.random() < self.arrival_prob:
            self.state += 1

        # 2. Simulate Service
        # Determine service rate based on action
        service_prob = self.service_rate_fast if action == 1 else self.service_rate_slow
        
        # If queue is not empty, attempt to serve a customer
        if self.state > 0 and random.random() < service_prob:
            self.state -= 1

        # 3. Apply Boundary Constraints (Queue cannot be < 0 or > max_queue)
        self.state = max(0, min(self.max_queue, self.state))

        # 4. Calculate Reward
        # Goal: Minimize Total Cost (Wait Cost + Operation Cost)
        # In RL, we maximize Reward, so Reward = -Cost
        wait_cost = self.state * self.cost_per_person
        operation_cost = self.cost_energy_fast if action == 1 else 0
        
        reward = -(wait_cost + operation_cost)

        return self.state, reward

# ==========================================
# Part 2: Agent Definition (The Self-Improving AI)
# ==========================================
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.9, epsilon=1.0):
        self.q_table = np.zeros((state_size, action_size)) # The "Brain" (Q-Table)
        self.lr = learning_rate      # Alpha: How much we accept new information
        self.gamma = gamma           # Discount Factor: How much we care about future rewards
        self.epsilon = epsilon       # Exploration Rate: Probability of choosing random action
        self.epsilon_decay = 0.995   # Decay factor to reduce exploration over time
        self.epsilon_min = 0.01      # Minimum exploration rate

    def choose_action(self, state):
        """Selects an action using Epsilon-Greedy Strategy."""
        # Exploration: Randomly try an action
        if random.random() < self.epsilon:
            return random.choice([0, 1])  
        # Exploitation: Choose the best known action
        else:
            return np.argmax(self.q_table[state]) 

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-Table using the Bellman Equation.
        Q(s,a) = Q(s,a) + lr * [R + gamma * max(Q(s')) - Q(s,a)]
        """
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Update the value in the table
        self.q_table[state, action] += self.lr * (target - predict)

    def decay_epsilon(self):
        """Reduces exploration rate as the agent becomes more experienced."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==========================================
# Part 3: Main Loop (Training & Evaluation)
# ==========================================
if __name__ == "__main__":
    # Initialization
    MAX_QUEUE_SIZE = 15
    env = ServiceQueueEnv(max_queue=MAX_QUEUE_SIZE)
    agent = QLearningAgent(state_size=MAX_QUEUE_SIZE + 1, action_size=2)

    episodes = 500
    scores = []

    print("🚀 Starting Self-Improving Training Process...")
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        # Simulate 100 minutes per episode
        for time in range(100):
            action = agent.choose_action(state)
            next_state, reward = env.step(action)
            
            # The "Self-Improving" Step: Agent reflects and updates its policy
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
        
        # Decay exploration rate after each episode
        agent.decay_epsilon()
        scores.append(total_reward)
        
        if (e+1) % 50 == 0:
            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    print("\n✅ Training Complete!")
    
    # ---------------------------
    # Result Visualization: Learning Curve
    # ---------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='Reward per Episode')
    plt.title('Self-Improving Process: Learning Curve')
    plt.xlabel('Episode (Experience)')
    plt.ylabel('Total Reward (Higher is Better)')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png') 
    print("📊 Learning curve saved as 'learning_curve.png'")

    # ---------------------------
    # Result Analysis: Final Policy
    # ---------------------------
    print("\n🧠 Final Policy Learned by AI (Q-Table Analysis):")
    print(f"{'Queue Length':<15} | {'Best Action':<15} | {'Interpretation'}")
    print("-" * 55)
    for s in range(MAX_QUEUE_SIZE + 1):
        best_action = np.argmax(agent.q_table[s])
        action_str = "🚀 Fast Mode" if best_action == 1 else "🐢 Slow Mode"
        interpret = "High cost, clear queue" if best_action == 1 else "Save energy"
        print(f"{s:<15} | {action_str:<15} | {interpret}")