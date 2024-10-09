'''
prioritized experience replay buffer
using for discret action spaces algorithms:
{DQN, DDQN...}
author : Belabed Abdelkader

'''

import torch
import numpy as np 


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.max_priority = 1.0
        self.index = 0

    def push(self, state, action, reward, next_state, done):
        if state is None or next_state is None:
            return  

        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(float(self.max_priority))  # Ensure the priority is a float
        else:
            self.buffer[self.index] = experience
            self.priorities[self.index] = float(self.max_priority)  # Ensure the priority is a float

        self.index = (self.index + 1) % self.capacity



    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            print("Replay buffer is empty!")
            return None  

        # Convert priorities to a numpy array ensuring it contains only floats
        try:
            priorities = np.array([float(p) for p in self.priorities], dtype=np.float32)
        except ValueError as e:
            print("Error converting priorities to numpy array:", e)
            return None

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        if any(sample is None for sample in samples):
            print("Found None in samples!")
            return None  

        states, actions, rewards, next_states, dones = zip(*samples)

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights).float()

        return (
            torch.tensor(np.array(states)).float(),
            torch.tensor(np.array(actions)).long(),
            torch.tensor(np.array(rewards)).unsqueeze(1).float(),
            torch.tensor(np.array(next_states)).float(),
            torch.tensor(np.array(dones)).unsqueeze(1).int(),
            weights
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority)  # Ensure the priority is a float
            self.max_priority = max(self.max_priority, priority)


    def __len__(self):
        return len(self.buffer)
