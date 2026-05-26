#!/usr/bin/env python3
"""
trainer.py — PPO training loop for the ML scheduler.
Connects: FeatureExtractor → SchedulerNetwork → RewardComputer
"""
import os
import sys
import time
import signal
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.extractor import FeatureExtractor
from agent.network       import SchedulerNetwork, ACTION_DIM
from agent.rollout       import RolloutBuffer
from agent.reward        import RewardComputer

# ── Hyperparameters ───────────────────────────────────────────
STEPS_PER_EPISODE = 64       # steps before each PPO update
PPO_EPOCHS        = 4        # how many passes over each rollout
CLIP_EPS          = 0.2      # PPO clip range
GAMMA             = 0.99     # discount factor
GAE_LAMBDA        = 0.95     # GAE smoothing
LR                = 3e-4     # learning rate
ENTROPY_COEFF     = 0.01     # encourages exploration
VALUE_COEFF       = 0.5      # critic loss weight
STEP_INTERVAL     = 0.5      # seconds between steps
SAVE_EVERY        = 10       # save model every N episodes

# ── Action application ────────────────────────────────────────
def apply_action(pid: int, action: int):
    """
    Apply scheduling decision to a real process.
    Uses os.setpriority — affects real kernel scheduler.
    Safe range: nice -5 to +5 only (conservative for now)
    """
    try:
        current_nice = os.getpriority(os.PRIO_PROCESS, pid)

        if action == 0:   # lower priority
            new_nice = min(current_nice + 5, 19)
            os.setpriority(os.PRIO_PROCESS, pid, new_nice)

        elif action == 1:  # raise priority
            new_nice = max(current_nice - 5, -5)
            os.setpriority(os.PRIO_PROCESS, pid, new_nice)

        elif action == 2:  # keep — no change
            pass

        elif action == 3:  # SCHED_BATCH hint
            # Mark as batch process via renice to positive
            new_nice = min(current_nice + 3, 19)
            os.setpriority(os.PRIO_PROCESS, pid, new_nice)

        elif action == 4:  # SCHED_OTHER — reset to normal
            os.setpriority(os.PRIO_PROCESS, pid, 0)

    except (ProcessLookupError, PermissionError):
        pass   # process died or no permission — skip silently

# ── PID management ────────────────────────────────────────────
def get_active_pids(min_count: int = 5) -> list:
    """
    Get currently active non-system PIDs to schedule.
    Avoids PID 0 (idle) and PID 1 (init) for safety.
    """
    pids = []
    try:
        for entry in os.scandir('/proc'):
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid <= 2:
                continue
            # Check process is still alive
            try:
                os.kill(pid, 0)
                pids.append(pid)
            except:
                continue
    except:
        pass
    return pids if len(pids) >= min_count else []

# ── PPO Update ────────────────────────────────────────────────
def ppo_update(network, optimizer, buffer):
    """
    Core PPO update step.
    Called once per episode after rollout is complete.
    """
    states, actions, old_log_probs, advantages, returns = \
        buffer.compute_returns(GAMMA, GAE_LAMBDA)

    total_loss_sum = 0.0

    for epoch in range(PPO_EPOCHS):
        # Evaluate current policy on collected experience
        new_log_probs, values, entropy = network.evaluate(states, actions)

        # Probability ratio — new policy vs old policy
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
        actor_loss  = -torch.min(surr1, surr2).mean()

        # Critic loss — how wrong was the value estimate?
        critic_loss = nn.MSELoss()(values, returns)

        # Entropy bonus — prevents premature convergence
        entropy_loss = -entropy.mean()

        # Combined loss
        loss = (actor_loss
                + VALUE_COEFF  * critic_loss
                + ENTROPY_COEFF * entropy_loss)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(network.parameters(), 0.5)
        optimizer.step()

        total_loss_sum += loss.item()

    return total_loss_sum / PPO_EPOCHS

# ── Main Training Loop ────────────────────────────────────────
def train():
    print("Initialising ML Scheduler Trainer...")

    # Components
    network   = SchedulerNetwork()
    optimizer = optim.Adam(network.parameters(), lr=LR)
    extractor = FeatureExtractor()
    buffer    = RolloutBuffer()

    # Track PIDs for reward — top active processes
    import subprocess
    result = subprocess.run(
        ['ps', '--no-headers', '-eo', 'pid', '--sort=-%cpu'],
        capture_output=True, text=True
    )
    reward_pids = []
    for line in result.stdout.strip().split('\n')[:20]:
        try:
            reward_pids.append(int(line.strip()))
        except:
            continue

    reward_computer = RewardComputer(reward_pids)
    reward_computer.compute()  # establish baseline

    # Training state
    episode       = 0
    total_steps   = 0
    running       = True
    episode_rewards = []

    def shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"Tracking {len(reward_pids)} PIDs for reward")
    print(f"Steps per episode: {STEPS_PER_EPISODE}")
    print(f"Starting training... (Ctrl+C to stop)\n")

    while running:
        episode += 1
        buffer.clear()
        step_rewards = []

        # ── Rollout collection ────────────────────────────────
        for step in range(STEPS_PER_EPISODE):
            if not running:
                break

            # Pick a random active PID to make decision for
            active_pids = get_active_pids()
            if not active_pids:
                time.sleep(STEP_INTERVAL)
                continue

            pid = random.choice(active_pids)

            # Get state
            state_np = extractor.extract(pid)
            state    = torch.FloatTensor(state_np).unsqueeze(0)

            # Network decision
            with torch.no_grad():
                action, log_prob, value = network.act(state)

            # Apply to real process
            apply_action(pid, action.item())

            # Wait and measure reward
            time.sleep(STEP_INTERVAL)
            reward = reward_computer.compute()

            # Store experience
            buffer.add(
                state    = state_np,
                action   = action.item(),
                reward   = reward,
                log_prob = log_prob.item(),
                value    = value.item(),
                done     = (step == STEPS_PER_EPISODE - 1),
            )

            step_rewards.append(reward)
            total_steps += 1

        # ── PPO Update ────────────────────────────────────────
        if len(buffer) < 2:
            continue

        avg_loss   = ppo_update(network, optimizer, buffer)
        avg_reward = np.mean(step_rewards)
        episode_rewards.append(avg_reward)

        print(f"Episode {episode:4d} | "
              f"Steps {total_steps:6d} | "
              f"Avg Reward {avg_reward:+.4f} | "
              f"Loss {avg_loss:.4f} | "
              f"Buffer {len(buffer)}")

        # ── Save checkpoint ───────────────────────────────────
        if episode % SAVE_EVERY == 0:
            path = f"checkpoints/scheduler_ep{episode}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'episode':     episode,
                'model':       network.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'rewards':     episode_rewards,
            }, path)
            print(f"  → Saved checkpoint: {path}")

    print("\nTraining stopped.")
    print(f"Total episodes: {episode}")
    print(f"Total steps:    {total_steps}")

if __name__ == "__main__":
    train()
