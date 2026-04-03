"""
Reward functions for the countdown task.
"""

import re


def countdown_accuracy_reward(completions, target, nums, **kwargs):
    """Binary accuracy reward: 1.0 if correct, 0.0 otherwise."""
    rewards = []
    for completion, t, n in zip(completions, target, nums):
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = completion

        # Extract answer from <answer> tags
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        equation = match.group(1).strip()
        try:
            # Validate only uses provided numbers
            used_numbers = [int(x) for x in re.findall(r'\d+', equation)]
            available = list(n)
            valid = True
            for num in used_numbers:
                if num in available:
                    available.remove(num)
                else:
                    valid = False
                    break

            if not valid:
                rewards.append(0.0)
                continue

            # Check only allowed characters
            if not re.match(r'^[\d+\-*/().\s]+$', equation):
                rewards.append(0.0)
                continue

            result = eval(equation)
            if abs(result - t) < 1e-6:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)

    return rewards