#!/usr/bin/env python3
"""
export_weights.py
Exports trained PyTorch model weights to a C header file.
The C inference engine uses this header — no PyTorch needed at runtime.
"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agent.network import SchedulerNetwork

# ── Config ────────────────────────────────────────────────────
CHECKPOINT = "checkpoints/scheduler_ep40.pt"
OUTPUT     = "kernel/ml_weights.h"

# ── Load model ────────────────────────────────────────────────
print(f"Loading checkpoint: {CHECKPOINT}")
checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)

network = SchedulerNetwork()
network.load_state_dict(checkpoint['model'])
network.eval()

print(f"Loaded. Episode: {checkpoint['episode']}")
print(f"Rewards history length: {len(checkpoint['rewards'])}")

# ── Helper — write one weight array ──────────────────────────
def write_array(f, c_name, tensor):
    data  = tensor.detach().numpy().flatten()
    shape = list(tensor.shape)
    f.write(f"/* {c_name}  shape: {shape}  "
            f"elements: {len(data)} */\n")
    f.write(f"static const float {c_name}[{len(data)}] = {{\n")

    # Write 8 values per line — readable
    for i in range(0, len(data), 8):
        chunk = data[i:i+8]
        line  = "    " + ", ".join(f"{v:.8f}f" for v in chunk)
        if i + 8 < len(data):
            line += ","
        f.write(line + "\n")

    f.write("};\n\n")

# ── Export ────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print(f"Exporting weights to: {OUTPUT}")

with open(OUTPUT, 'w') as f:
    f.write("/*\n")
    f.write(" * ml_weights.h — Auto-generated ML scheduler weights\n")
    f.write(f" * Source checkpoint: {CHECKPOINT}\n")
    f.write(f" * Episode: {checkpoint['episode']}\n")
    f.write(f" * Do NOT edit manually — regenerate via export_weights.py\n")
    f.write(" */\n\n")
    f.write("#ifndef ML_WEIGHTS_H\n")
    f.write("#define ML_WEIGHTS_H\n\n")
    f.write("/* Network dimensions */\n")
    f.write("#define ML_INPUT_DIM   10\n")
    f.write("#define ML_HIDDEN_DIM  64\n")
    f.write("#define ML_ACTION_DIM   5\n\n")

    # Export every layer
    params = dict(network.named_parameters())

    # Shared layer 1
    write_array(f, "shared_0_weight",
                params['shared.0.weight'])
    write_array(f, "shared_0_bias",
                params['shared.0.bias'])

    # Shared layer 2
    write_array(f, "shared_2_weight",
                params['shared.2.weight'])
    write_array(f, "shared_2_bias",
                params['shared.2.bias'])

    # Actor head
    write_array(f, "actor_weight",
                params['actor.weight'])
    write_array(f, "actor_bias",
                params['actor.bias'])

    # Critic head
    write_array(f, "critic_weight",
                params['critic.weight'])
    write_array(f, "critic_bias",
                params['critic.bias'])

    f.write("#endif /* ML_WEIGHTS_H */\n")

print(f"Done. Verifying output...")

# ── Verify — run inference in Python and C will match ─────────
test_input = torch.FloatTensor(
    [[0.06, 0.04, 0.007, 0.09, 0.006,
      0.06, 0.14, 0.08,  0.06, 1.0]]
)
with torch.no_grad():
    logits, value = network(test_input)
    probs = torch.softmax(logits, dim=-1)

print(f"\nVerification — Python inference on test input:")
print(f"  Action probs: {probs.numpy().flatten().round(4)}")
print(f"  Value estimate: {value.item():.6f}")
print(f"  Chosen action: {probs.argmax().item()}")
print(f"\nSave these numbers — C inference must match exactly.")

# Count exported parameters
total = sum(p.numel() for p in network.parameters())
print(f"\nTotal parameters exported: {total}")
print(f"Output file: {OUTPUT}")
