import numpy as np
import torch

class RolloutBuffer:
    """
    Stores one episode of experience for PPO update.
    Cleared after every update — PPO is on-policy.
    """

    def __init__(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.log_probs = []
        self.values    = []
        self.dones     = []

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, gamma=0.99, gae_lambda=0.95):
        """
        Compute GAE advantages and discounted returns.
        
        GAE = Generalised Advantage Estimation
        Balances bias vs variance in advantage estimates.
        gamma      = how much to value future rewards (0.99 = care a lot)
        gae_lambda = smoothing factor (0.95 = standard)
        """
        advantages = []
        returns    = []
        gae        = 0

        # Work backwards through the rollout
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_value = 0  # episode ended
            else:
                next_value = self.values[step + 1]

            # TD error — how wrong was the critic?
            delta = (self.rewards[step]
                     + gamma * next_value * (1 - self.dones[step])
                     - self.values[step])

            # GAE accumulation
            gae = delta + gamma * gae_lambda * (1 - self.dones[step]) * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])

        # Normalise advantages — zero mean, unit variance
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return (
            torch.tensor(np.array(self.states),    dtype=torch.float32),
            torch.tensor(self.actions,             dtype=torch.long),
            torch.tensor(self.log_probs,           dtype=torch.float32),
            advantages,
            torch.tensor(returns,                  dtype=torch.float32),
        )

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)
