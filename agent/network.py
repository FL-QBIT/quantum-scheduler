import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Must match your feature extractor output exactly
STATE_DIM  = 10

# Must match your defined action space exactly
ACTION_DIM = 5

# Actions — keep these as constants so nothing is magic numbers
ACTION_LOWER_PRIO    = 0   # nice +5  (lower priority)
ACTION_RAISE_PRIO    = 1   # nice -5  (raise priority)
ACTION_KEEP_PRIO     = 2   # no change
ACTION_BATCH_POLICY  = 3   # SCHED_BATCH  (CPU bound hint)
ACTION_NORMAL_POLICY = 4   # SCHED_OTHER  (interactive hint)

class SchedulerNetwork(nn.Module):
    """
    Actor-Critic network for process scheduling.

    Takes a 10-element feature vector (one process's state)
    Returns:
      - action probabilities (actor)  → what to do
      - state value estimate (critic) → how good is this situation
    """

    def __init__(self):
        super().__init__()

        # Shared encoder — both actor and critic learn from this
        # This is the 10 → 64 → 64 path you drew
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, 64),   # 10 → 64
            nn.ReLU(),
            nn.Linear(64, 64),          # 64 → 64
            nn.ReLU(),
        )

        # Actor head — 64 → 5 (one logit per action)
        self.actor  = nn.Linear(64, ACTION_DIM)

        # Critic head — 64 → 1 (single value estimate)
        self.critic = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor):
        """
        Forward pass through the network.

        state: tensor of shape [batch_size, 10]
        returns: (action_logits, state_value)
        """
        # Pass through shared layers
        shared_out = self.shared(state)

        # Split into actor and critic heads
        action_logits = self.actor(shared_out)   # shape: [batch, 5]
        state_value   = self.critic(shared_out)  # shape: [batch, 1]

        return action_logits, state_value

    def act(self, state: torch.Tensor):
        """
        Given a state, sample an action from the policy.
        Used during rollout collection (interacting with environment).

        Returns:
          action      → the chosen action (0-4)
          log_prob    → log probability of that action (needed for PPO)
          value       → critic's estimate of state value
        """
        action_logits, value = self.forward(state)

        # Convert logits to probability distribution
        dist = Categorical(logits=action_logits)

        # Sample an action — stochastic, not greedy
        action = dist.sample()

        # Log probability — how likely was this action?
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate states and actions during PPO update.
        Used during training (not during rollout).

        Returns:
          log_probs → log probabilities of the taken actions
          values    → critic estimates
          entropy   → distribution entropy (encourages exploration)
        """
        action_logits, values = self.forward(states)
        dist = Categorical(logits=action_logits)

        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


#TEST
if __name__ == "__main__":
    # Instantiate the network
    net = SchedulerNetwork()
    print("Network architecture:")
    print(net)
    print()

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params}")
    print()

    # Fake a feature vector — same shape as your extractor output
    dummy_state = torch.FloatTensor(
        [0.06, 0.04, 0.007, 0.09, 0.006, 0.06, 0.14, 0.08, 0.06, 1.0]
    ).unsqueeze(0)  # unsqueeze adds batch dimension → shape [1, 10]

    # Forward pass
    logits, value = net.forward(dummy_state)
    print(f"Action logits: {logits}")
    print(f"State value:   {value}")
    print()

    # Act
    action, log_prob, val = net.act(dummy_state)
    print(f"Sampled action:  {action.item()} "
          f"({['lower','raise','keep','batch','normal'][action.item()]})")
    print(f"Log probability: {log_prob.item():.4f}")
    print(f"Value estimate:  {val.item():.4f}")
