#!/usr/bin/env python3
from bcc import BPF
import ctypes
import sys

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

    __builtin_memcpy(e.prev_comm, args->prev_comm, 16);
    __builtin_memcpy(e.next_comm, args->next_comm, 16);

    events.perf_submit(args, &e, sizeof(e));
    return 0;
}
"""

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
    ]

def decode_state(state):
    if state == 0:
        return "R"
    states = {
        0x01: "S",
        0x02: "D",
        0x04: "T",
        0x08: "t",
        0x10: "X",
        0x20: "Z",
    }
    result = [label for bit, label in states.items() if state & bit]
    return "|".join(result) if result else "R"

def handle_event(cpu, data, size):
    e = ctypes.cast(data, ctypes.POINTER(SchedEvent)).contents
    state = decode_state(e.prev_state)
    print(
        f"[{e.timestamp:>20}] "
        f"{e.prev_comm.decode():>16} ({e.prev_pid:>6}) "
        f"--[{state}]--> "
        f"{e.next_comm.decode():>16} ({e.next_pid:>6}) "
        f"prio={e.next_prio}"
    )

def main():
    print("Loading eBPF probe... (Ctrl+C to stop)")
    b = BPF(text=PROBE_CODE)
    b["events"].open_perf_buffer(handle_event)
    print("Watching scheduler events:\n")
    try:
        while True:
            b.perf_buffer_poll(timeout=100)
    except KeyboardInterrupt:
        print("\nDetaching probe...")
        sys.exit(0)

if __name__ == "__main__":
    main()