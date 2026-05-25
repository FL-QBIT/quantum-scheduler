#!/usr/bin/env python3
"""
monitor.py — connects eBPF probe to feature extractor.
Shows live feature vectors for active processes.
"""
from bcc import BPF
import ctypes
import sys
import os
import signal
import time
import threading
import numpy as np

# Add pipeline directory to path
sys.path.insert(0, os.path.dirname(__file__))
from extractor import FeatureExtractor

# ── eBPF probe code ───────────────────────────────────────────
PROBE_CODE = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct sched_event {
    u32 prev_pid;
    u32 next_pid;
    u64 timestamp;
    u32 prev_prio;
    u32 next_prio;
    long prev_state;
    char prev_comm[16];
    char next_comm[16];
    u32 cpu;
};

BPF_PERF_OUTPUT(events);

TRACEPOINT_PROBE(sched, sched_switch)
{
    struct sched_event e = {};
    e.prev_pid   = args->prev_pid;
    e.next_pid   = args->next_pid;
    e.prev_prio  = args->prev_prio;
    e.next_prio  = args->next_prio;
    e.prev_state = args->prev_state;
    e.timestamp  = bpf_ktime_get_ns();
    e.cpu        = bpf_get_smp_processor_id();
    __builtin_memcpy(e.prev_comm, args->prev_comm, 16);
    __builtin_memcpy(e.next_comm, args->next_comm, 16);
    events.perf_submit(args, &e, sizeof(e));
    return 0;
}
"""

# ── Python struct mirror ──────────────────────────────────────
class SchedEvent(ctypes.Structure):
    _fields_ = [
        ("prev_pid",   ctypes.c_uint32),
        ("next_pid",   ctypes.c_uint32),
        ("timestamp",  ctypes.c_uint64),
        ("prev_prio",  ctypes.c_uint32),
        ("next_prio",  ctypes.c_uint32),
        ("prev_state", ctypes.c_long),
        ("prev_comm",  ctypes.c_char * 16),
        ("next_comm",  ctypes.c_char * 16),
        ("cpu",        ctypes.c_uint32),
    ]

# ── Shared state ──────────────────────────────────────────────
extractor   = FeatureExtractor()
active_pids = set()   # PIDs seen in last interval
running     = True

# ── eBPF callback ─────────────────────────────────────────────
def handle_event(cpu, data, size):
    e = ctypes.cast(data, ctypes.POINTER(SchedEvent)).contents

    # Build event dict for extractor
    event = {
        'prev_pid':   e.prev_pid,
        'next_pid':   e.next_pid,
        'timestamp':  e.timestamp,
        'prev_state': e.prev_state,
        'cpu':        e.cpu,
        'prev_comm':  e.prev_comm.decode(errors='replace'),
    }

    # Feed into extractor's EventBuffer
    extractor.add_event(event)

    # Track active PIDs (ignore idle process PID 0)
    if e.prev_pid != 0:
        active_pids.add(e.prev_pid)

# ── Feature table printer (runs in background thread) ─────────
LABELS = [
    "cpu", "io", "wait", "burst",
    "slices", "mem", "prio", "threads",
    "state", "affinity"
]

def print_table():
    """Prints feature vectors for active PIDs every 3 seconds."""
    while running:
        time.sleep(3)
        if not active_pids:
            continue

        # Clear screen
        os.system('clear')
        print(f"{'PID':>8} {'COMM':>18}  " + "  ".join(f"{l:>7}" for l in LABELS))
        print("─" * 120)

        # Snapshot active PIDs to avoid set-change-during-iteration
        snapshot = list(active_pids)[:20]  # show top 20

        for pid in snapshot:
            try:
                # Get process name
                with open(f'/proc/{pid}/comm') as f:
                    comm = f.read().strip()[:18]
            except:
                active_pids.discard(pid)  # remove dead PIDs
                continue  

            features = extractor.extract(pid)
            vals = "  ".join(f"{v:>7.4f}" for v in features)
            print(f"{pid:>8} {comm:>18}  {vals}")

        print(f"\nTracking {len(active_pids)} PIDs  |  {time.strftime('%H:%M:%S')}")

# ── Signal handler ────────────────────────────────────────────
def shutdown(sig, frame):
    global running
    running = False

# ── Main ──────────────────────────────────────────────────────
def main():
    global running

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Loading eBPF probe...")
    b = BPF(text=PROBE_CODE)
    b["events"].open_perf_buffer(handle_event)
    print("Probe loaded. Watching scheduler...\n")

    # Start printer in background thread
    printer = threading.Thread(target=print_table, daemon=True)
    printer.start()

    # Poll eBPF events in main thread
    while running:
        b.perf_buffer_poll(timeout=200)

    print("\nDetaching probe...")
    b.cleanup()

if __name__ == "__main__":
    main()
