#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

struct sched_event {
    __u32 prev_pid;
    __u32 next_pid;
    __u64 timestamp;
    __u32 prev_prio;
    __u32 next_prio;
    long  prev_state;
    char  prev_comm[16];
    char  next_comm[16];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24);
} events SEC(".maps");

SEC("tp/sched/sched_switch")
int handle_sched_switch(struct trace_event_raw_sched_switch *ctx)
{
    struct sched_event *e;

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->prev_pid   = ctx->prev_pid;
    e->next_pid   = ctx->next_pid;
    e->prev_prio  = ctx->prev_prio;
    e->next_prio  = ctx->next_prio;
    e->prev_state = ctx->prev_state;
    e->timestamp  = bpf_ktime_get_ns();

    __builtin_memcpy(e->prev_comm, ctx->prev_comm, 16);
    __builtin_memcpy(e->next_comm, ctx->next_comm, 16);

    bpf_ringbuf_submit(e, 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";