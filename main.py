import numpy as np
import matplotlib.pyplot as plt
import time

# ==================================================
# MDP PARAMETERS
# ==================================================

N_max = 15
states = list(range(N_max + 1))
actions = [0, 1]  # 0 = Normal, 1 = Fast

lambda_arrival = 0.3
mu_normal = 0.4
mu_fast = 0.7

gamma = 0.95

holding_cost_coeff = 2
service_cost = {0: 1, 1: 5}

theta = 1e-8
max_eval_iterations = 1000

# ==================================================
# Transition Function
# ==================================================

def get_transition_probs(s, a):
    probs = []
    mu = mu_normal if a == 0 else mu_fast

    arrival_prob = lambda_arrival
    service_prob = mu

    stay_prob = 1 - arrival_prob - service_prob

    # Numerical safety
    stay_prob = max(0.0, stay_prob)

    # Arrival
    if s < N_max:
        probs.append((arrival_prob, s + 1))
    else:
        probs.append((arrival_prob, s))

    # Service
    if s > 0:
        probs.append((service_prob, s - 1))
    else:
        probs.append((service_prob, s))

    # Stay
    probs.append((stay_prob, s))

    return probs


# ==================================================
# Reward Function
# ==================================================

def reward(s, a):
    holding_cost = holding_cost_coeff * s
    return -(holding_cost + service_cost[a])


# ==================================================
# Policy Evaluation
# ==================================================

def policy_evaluation(policy):
    V = np.zeros(len(states))

    for _ in range(max_eval_iterations):
        delta = 0
        new_V = np.copy(V)

        for s in states:
            a = policy[s]
            value = 0

            for prob, s_next in get_transition_probs(s, a):
                value += prob * (reward(s, a) + gamma * V[s_next])

            new_V[s] = value
            delta = max(delta, abs(V[s] - new_V[s]))

        V = new_V

        if delta < theta:
            break

    return V


# ==================================================
# Policy Improvement
# ==================================================

def policy_improvement(policy, V):
    policy_stable = True
    new_policy = np.copy(policy)

    for s in states:
        old_action = policy[s]
        action_values = []

        for a in actions:
            q = 0
            for prob, s_next in get_transition_probs(s, a):
                q += prob * (reward(s, a) + gamma * V[s_next])
            action_values.append(q)

        best_action = np.argmax(action_values)
        new_policy[s] = best_action

        if best_action != old_action:
            policy_stable = False

    return new_policy, policy_stable


# ==================================================
# Policy Iteration
# ==================================================

def policy_iteration():
    policy = np.zeros(len(states), dtype=int)

    iteration = 0
    start_time = time.time()

    while True:
        iteration += 1

        V = policy_evaluation(policy)
        new_policy, stable = policy_improvement(policy, V)

        if stable:
            break

        policy = new_policy

    end_time = time.time()

    print("====================================")
    print("Policy Iteration Converged")
    print("Iterations:", iteration)
    print("Time (seconds):", round(end_time - start_time, 4))
    print("====================================")

    return policy, V


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    optimal_policy, optimal_value = policy_iteration()

    print("\nOptimal Policy:")
    for s in states:
        action_name = "FAST" if optimal_policy[s] == 1 else "NORMAL"
        print(f"Queue length {s}: {action_name}")

    # ==================================================
    # Plot Value Function
    # ==================================================

    plt.figure()
    plt.plot(states, optimal_value)
    plt.xlabel("Queue Length")
    plt.ylabel("Value Function")
    plt.title("Optimal Value Function")
    plt.grid(True)
    plt.show()

    # ==================================================
    # Plot Policy
    # ==================================================

    plt.figure()
    plt.plot(states, optimal_policy)
    plt.xlabel("Queue Length")
    plt.ylabel("Action (0=Normal, 1=Fast)")
    plt.title("Optimal Policy")
    plt.grid(True)
    plt.show()