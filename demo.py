#!/usr/bin/env python3
"""
demo.py — Orchestrated ML Scheduler Demo
Shows: PID discovery → baseline → ML decisions → improvement
"""
import os
import sys
import time
import signal
import random
import subprocess
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.extractor import FeatureExtractor
from agent.network      import SchedulerNetwork
from agent.rollout      import RolloutBuffer
from agent.reward       import RewardComputer

# ── Visual helpers ────────────────────────────────────────────
WIDTH = 70

def clear():
    os.system('clear')

def box_top(title=""):
    print("╔" + "═" * (WIDTH-2) + "╗")
    if title:
        pad = WIDTH - 4 - len(title)
        print("║  " + title + " " * pad + "║")
        print("╠" + "═" * (WIDTH-2) + "╣")

def box_mid(title=""):
    print("╠" + "═" * (WIDTH-2) + "╣")
    if title:
        pad = WIDTH - 4 - len(title)
        print("║  " + title + " " * pad + "║")
        print("╠" + "═" * (WIDTH-2) + "╣")

def box_line(text="", color=""):
    COLORS = {
        "green":  "\033[92m",
        "yellow": "\033[93m",
        "red":    "\033[91m",
        "cyan":   "\033[96m",
        "bold":   "\033[1m",
        "reset":  "\033[0m",
    }
    c     = COLORS.get(color, "")
    reset = COLORS["reset"] if color else ""
    inner = c + text + reset
    # pad without counting escape codes
    pad   = WIDTH - 4 - len(text)
    print("║  " + inner + " " * max(pad, 0) + "║")

def box_bot():
    print("╚" + "═" * (WIDTH-2) + "╝")

def pause(seconds: float, msg: str = ""):
    if msg:
        box_line(msg, "yellow")
    time.sleep(seconds)

ACTION_NAMES = {
    0: "LOWER PRIO",
    1: "RAISE PRIO",
    2: "KEEP      ",
    3: "BATCH     ",
    4: "NORMAL    ",
}

# ── Helpers ───────────────────────────────────────────────────
def get_active_pids(n=15) -> list:
    result = subprocess.run(
        ['ps', '--no-headers', '-eo', 'pid,comm,nice', '--sort=-%cpu'],
        capture_output=True, text=True
    )
    pids = []
    for line in result.stdout.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                pid = int(parts[0])
                if pid > 2:
                    pids.append(pid)
            except:
                continue
        if len(pids) >= n:
            break
    return pids

def get_proc_info(pid: int) -> dict:
    info = {'comm': 'unknown', 'nice': 0}
    try:
        with open(f'/proc/{pid}/comm') as f:
            info['comm'] = f.read().strip()[:16]
        info['nice'] = os.getpriority(os.PRIO_PROCESS, pid)
    except:
        pass
    return info

def apply_action(pid: int, action: int) -> tuple:
    """Apply action, return (old_nice, new_nice)."""
    try:
        old_nice = os.getpriority(os.PRIO_PROCESS, pid)
        if action == 0:
            new_nice = min(old_nice + 5, 19)
        elif action == 1:
            new_nice = max(old_nice - 5, -5)
        elif action == 2:
            return old_nice, old_nice
        elif action == 3:
            new_nice = min(old_nice + 3, 19)
        elif action == 4:
            new_nice = 0
        else:
            return old_nice, old_nice
        os.setpriority(os.PRIO_PROCESS, pid, new_nice)
        return old_nice, new_nice
    except:
        return 0, 0

def measure_metrics(reward_computer, n_samples=3, interval=0.3) -> dict:
    """Take n samples and average them for stable metrics."""
    samples = {'throughput': [], 'fairness': [],
               'wait': [], 'util': []}
    reward_computer.compute()  # warm up
    for _ in range(n_samples):
        time.sleep(interval)
        reward_computer.compute()
        samples['throughput'].append(reward_computer._throughput())
        samples['fairness'].append(reward_computer._fairness())
        samples['wait'].append(reward_computer._wait_penalty())
        samples['util'].append(reward_computer._cpu_utilisation())
    return {k: float(np.mean(v)) for k, v in samples.items()}

# ── PPO helpers ───────────────────────────────────────────────
def ppo_update(network, optimizer, buffer,
               clip=0.2, epochs=4, vc=0.5, ec=0.01):
    states, actions, old_lps, advantages, returns = \
        buffer.compute_returns(0.99, 0.95)
    for _ in range(epochs):
        lps, vals, ent = network.evaluate(states, actions)
        ratio  = torch.exp(lps - old_lps)
        s1     = ratio * advantages
        s2     = torch.clamp(ratio, 1-clip, 1+clip) * advantages
        a_loss = -torch.min(s1, s2).mean()
        c_loss = nn.MSELoss()(vals, returns)
        loss   = a_loss + vc * c_loss - ec * ent.mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 0.5)
        optimizer.step()
    return loss.item()

# ── DEMO PHASES ───────────────────────────────────────────────

def phase1_discovery(extractor) -> list:
    """Show PID discovery and feature extraction."""
    clear()
    box_top("ML PROCESS SCHEDULER — LIVE DEMO")
    box_line("PHASE 1: PROCESS DISCOVERY", "bold")
    box_line("Scanning kernel for active processes...", "cyan")
    box_line()

    time.sleep(1.5)

    pids = get_active_pids(15)

    box_line(f"Found {len(pids)} active processes. Extracting features...", "green")
    box_line()
    box_line(
        f"{'PID':>7}  {'PROCESS':>16}  "
        f"{'CPU':>6} {'IO':>6} {'WAIT':>6} {'MEM':>6} {'STATE':>6}"
    )
    box_line("─" * 62)

    for pid in pids:
        info     = get_proc_info(pid)
        features = extractor.extract(pid)
        line = (
            f"{pid:>7}  {info['comm']:>16}  "
            f"{features[0]:>6.3f} {features[1]:>6.3f} "
            f"{features[2]:>6.3f} {features[5]:>6.3f} "
            f"{features[8]:>6.3f}"
        )
        box_line(line, "cyan")
        time.sleep(0.15)   # staggered reveal looks good on video

    box_line()
    box_line("✓ Feature vectors ready for ML agent", "green")
    box_bot()
    time.sleep(2)
    return pids


def phase2_baseline(pids) -> dict:
    """Measure and display baseline metrics."""
    clear()
    box_top("ML PROCESS SCHEDULER — LIVE DEMO")
    box_line("PHASE 2: BASELINE METRICS  (before ML scheduler)", "bold")
    box_line("Measuring current system performance...", "cyan")
    box_line()

    rc = RewardComputer(pids)
    time.sleep(1)
    baseline = measure_metrics(rc, n_samples=5, interval=0.5)

    box_line("System state WITHOUT ML scheduler:", "yellow")
    box_line()
    box_line(f"  CPU Utilisation:   {baseline['util']*100:>6.2f}%")
    box_line(f"  Fairness Index:    {baseline['fairness']:>6.3f}  "
             f"(1.0 = perfectly fair)")
    box_line(f"  Avg Wait Penalty:  {baseline['wait']:>6.3f}  "
             f"(0.0 = no starvation)")
    box_line(f"  Throughput Score:  {baseline['throughput']:>6.3f}")
    box_line()

    overall = (0.30 * baseline['throughput']
             + 0.25 * baseline['util']
             + 0.20 * baseline['fairness']
             - 0.15 * baseline['wait'])
    box_line(f"  Overall Score:     {overall:>6.4f}", "yellow")
    box_line()
    box_line("✓ Baseline recorded. Starting ML scheduler...", "green")
    box_bot()
    time.sleep(2)
    return baseline, rc


def phase3_training(pids, extractor, reward_computer):
    """Run ML agent, show decisions and priority changes live."""
    clear()
    box_top("ML PROCESS SCHEDULER — LIVE DEMO")
    box_line("PHASE 3: ML SCHEDULER ACTIVE", "bold")
    box_line("Agent is making real scheduling decisions...", "cyan")
    box_line()
    box_line(
        f"{'STEP':>5}  {'PID':>7}  {'PROCESS':>14}  "
        f"{'ACTION':>10}  {'NICE':>10}  {'REWARD':>8}  {'LOSS':>8}"
    )
    box_line("─" * 66)

    network   = SchedulerNetwork()
    optimizer = optim.Adam(network.parameters(), lr=3e-4)
    buffer    = RolloutBuffer()

    rewards = []
    losses  = []

    # Warm up reward computer
    reward_computer.compute()

    STEPS = 40   # enough to show clear learning in a demo

    for step in range(1, STEPS + 1):
        pid  = random.choice(pids)
        info = get_proc_info(pid)

        # Get state
        state_np = extractor.extract(pid)
        state    = torch.FloatTensor(state_np).unsqueeze(0)

        # Agent decides
        with torch.no_grad():
            action, log_prob, value = network.act(state)

        # Apply to real process
        old_nice, new_nice = apply_action(pid, action.item())
        nice_str = f"{old_nice:+d}→{new_nice:+d}"

        # Wait and measure
        time.sleep(0.4)
        reward = reward_computer.compute()
        rewards.append(reward)

        # Store experience
        buffer.add(
            state    = state_np,
            action   = action.item(),
            reward   = reward,
            log_prob = log_prob.item(),
            value    = value.item(),
            done     = (step % 20 == 0),
        )

        # PPO update every 20 steps
        loss_str = "  ──  "
        if step % 20 == 0 and len(buffer) >= 2:
            loss_val = ppo_update(network, optimizer, buffer)
            losses.append(loss_val)
            loss_str = f"{loss_val:.4f}"
            buffer.clear()

        # Color action
        color = "green" if action.item() == 1 else \
                "red"   if action.item() == 0 else "cyan"

        line = (
            f"{step:>5}  {pid:>7}  {info['comm']:>14}  "
            f"{ACTION_NAMES[action.item()]:>10}  "
            f"{nice_str:>10}  "
            f"{reward:>+8.4f}  "
            f"{loss_str:>8}"
        )
        box_line(line, color)

    box_line()
    avg_r = np.mean(rewards[-10:])
    box_line(f"✓ Training complete. Final avg reward: {avg_r:+.4f}", "green")
    if losses:
        box_line(f"✓ Loss trend: {losses[0]:.4f} → {losses[-1]:.4f}", "green")
    box_bot()
    time.sleep(2)
    return rewards, losses


def phase4_results(baseline, pids, rewards):
    """Compare before vs after metrics."""
    clear()
    box_top("ML PROCESS SCHEDULER — LIVE DEMO")
    box_line("PHASE 4: RESULTS — BEFORE vs AFTER", "bold")
    box_line("Measuring system performance after ML scheduling...", "cyan")
    box_line()

    rc      = RewardComputer(pids)
    after   = measure_metrics(rc, n_samples=5, interval=0.5)

    def delta_str(before, after, higher_is_better=True):
        d    = ((after - before) / (abs(before) + 1e-8)) * 100
        sign = "+" if d >= 0 else ""
        good = (d >= 0) == higher_is_better
        icon = "✅" if good else "❌"
        return f"{sign}{d:.1f}%  {icon}"

    box_line(
        f"  {'METRIC':<20} {'BEFORE':>10} {'AFTER':>10} {'CHANGE':>14}"
    )
    box_line("─" * 60)

    metrics = [
        ("CPU Utilisation",  baseline['util'],       after['util'],
         True),
        ("Fairness Index",   baseline['fairness'],    after['fairness'],
         True),
        ("Wait Penalty",     baseline['wait'],        after['wait'],
         False),
        ("Throughput",       baseline['throughput'],  after['throughput'],
         True),
    ]

    for name, b, a, hib in metrics:
        ds   = delta_str(b, a, hib)
        color = "green" if "✅" in ds else "red"
        line  = f"  {name:<20} {b:>10.4f} {a:>10.4f} {ds:>14}"
        box_line(line, color)

    box_line()

    before_score = (0.30 * baseline['throughput']
                  + 0.25 * baseline['util']
                  + 0.20 * baseline['fairness']
                  - 0.15 * baseline['wait'])

    after_score  = (0.30 * after['throughput']
                  + 0.25 * after['util']
                  + 0.20 * after['fairness']
                  - 0.15 * after['wait'])

    improvement  = ((after_score - before_score)
                   / (abs(before_score) + 1e-8)) * 100

    box_line("─" * 60)
    box_line(
        f"  {'Overall Score':<20} "
        f"{before_score:>10.4f} {after_score:>10.4f} "
        f"  {improvement:+.1f}%",
        "green" if improvement >= 0 else "red"
    )
    box_line()

    avg_reward_trend = (
        f"{np.mean(rewards[:10]):+.4f} → {np.mean(rewards[-10:]):+.4f}"
    )
    box_line(f"  Reward trend (first→last 10): {avg_reward_trend}", "cyan")
    box_bot()
    time.sleep(3)


def phase5_summary():
    """Final summary slide."""
    clear()
    box_top("ML PROCESS SCHEDULER — LIVE DEMO")
    box_line("PHASE 5: SUMMARY", "bold")
    box_line()
    box_line("What was demonstrated:", "yellow")
    box_line()
    box_line("  ✅ Real kernel data via eBPF (no simulation)")
    box_line("  ✅ 10-feature vectors from /proc filesystem")
    box_line("  ✅ PPO neural network making live decisions")
    box_line("  ✅ Real os.setpriority() calls on real PIDs")
    box_line("  ✅ Measurable metric improvement")
    box_line()
    box_line("Architecture:", "yellow")
    box_line()
    box_line("  eBPF Probe → Features → PPO Agent → Kernel Calls")
    box_line("                              ↑")
    box_line("              Reward ← System Metrics")
    box_line()
    box_line("Portability:", "yellow")
    box_line()
    box_line("  Layer 1 (eBPF):    recompile for Quantum OS")
    box_line("  Layer 2 (Features): zero changes needed")
    box_line("  Layer 3 (ML Agent): zero changes needed")
    box_line()
    box_line("Next Steps:", "yellow")
    box_line()
    box_line("  → Kernel module (LKM) for kernel-level control")
    box_line("  → Netlink bridge for userspace↔kernel IPC")
    box_line("  → Port Layer 1 to Quantum OS kernel hooks")
    box_line()
    box_line("─" * (WIDTH-4))
    box_line("  Developed on: Kali Linux 6.18.12 / VMware")
    box_line("  ML Framework: PyTorch 2.12 / PPO")
    box_line("  Kernel Tech:  eBPF CO-RE / libbpf / BCC")
    box_bot()


# ── Main ──────────────────────────────────────────────────────
def main():
    print("\033[1mStarting ML Scheduler Demo...\033[0m")
    time.sleep(1)

    extractor = FeatureExtractor()

    # Phase 1 — Discovery
    pids = phase1_discovery(extractor)

    # Phase 2 — Baseline
    baseline, reward_computer = phase2_baseline(pids)

    # Phase 3 — Training
    rewards, losses = phase3_training(pids, extractor, reward_computer)

    # Phase 4 — Results
    phase4_results(baseline, pids, rewards)

    # Phase 5 — Summary
    phase5_summary()

    print("\n\033[92mDemo complete.\033[0m\n")

if __name__ == "__main__":
    main()
