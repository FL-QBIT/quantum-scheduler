import collections
import numpy as np

class EventBuffer:
    """
    Stores recent eBPF scheduler events per PID.
    Acts as the memory of what each process has been doing.
    """

    def __init__(self, maxlen=50):
        # deque = efficient fixed-size queue
        # when full, oldest event dropped automatically
        self.buffers = collections.defaultdict(
            lambda: collections.deque(maxlen=maxlen)
        )

    def add_event(self, event: dict):
        """Called by eBPF callback for every context switch."""
        pid = event['prev_pid']
        self.buffers[pid].append(event)

    def get_events(self, pid: int) -> list:
        return list(self.buffers.get(pid, []))

    def get_state_pattern(self, pid: int) -> float:
        """
        Returns a number 0.0 → 1.0 describing recent behaviour.
        0.0 = purely I/O bound (all D/S states)
        1.0 = purely CPU bound (all R states)
        0.5 = mixed / interactive
        """
        events = self.get_events(pid)
        if not events:
            return 0.5  # unknown → assume interactive

        cpu_bound_count = sum(1 for e in events if e['prev_state'] == 0)
        return cpu_bound_count / len(events)

    def get_cpu_affinity_score(self, pid: int) -> float:
        """
        1.0 = always on same CPU core (cache warm, good)
        0.0 = jumping between all cores (cache cold, bad)
        """
        events = self.get_events(pid)
        if not events:
            return 1.0

        cores_used = len(set(e['cpu'] for e in events))
        total_cores = max(cores_used, 1)
        # fewer cores used = better score
        return 1.0 / total_cores

import os

class ProcReader:
    """
    Reads /proc/<pid>/ files safely.
    Always handles missing files — processes can die at any moment.
    """

    def __init__(self):
        # clock ticks per second — needed to convert stat times
        self.clock_ticks = os.sysconf('SC_CLK_TCK')  # usually 100
        self.total_ram_kb = self._get_total_ram()

    def _get_total_ram(self) -> int:
        """Read total system RAM from /proc/meminfo."""
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal'):
                        return int(line.split()[1])
        except:
            return 8 * 1024 * 1024  # fallback: assume 8GB

    def read_stat(self, pid: int) -> dict:
        """
        /proc/<pid>/stat — CPU times, priority, state.
        Returns empty dict if process is gone.
        """
        try:
            with open(f'/proc/{pid}/stat') as f:
                fields = f.read().split()
            return {
                'state':    fields[2],
                'utime':    int(fields[13]),  # user CPU ticks
                'stime':    int(fields[14]),  # kernel CPU ticks
                'priority': int(fields[17]),
                'nice':     int(fields[18]),
                'threads':  int(fields[19]),
            }
        except:
            return {}

    def read_schedstat(self, pid: int) -> dict:
        """
        /proc/<pid>/schedstat — wait time, run time, timeslices.
        This is the most valuable file for scheduler decisions.
        """
        try:
            with open(f'/proc/{pid}/schedstat') as f:
                cpu_time, wait_time, slices = map(int, f.read().split())
            return {
                'cpu_time_ns':  cpu_time,
                'wait_time_ns': wait_time,
                'timeslices':   slices,
            }
        except:
            return {}

    def read_io(self, pid: int) -> dict:
        """
        /proc/<pid>/io — bytes read and written.
        High values = I/O bound process.
        """
        try:
            result = {}
            with open(f'/proc/{pid}/io') as f:
                for line in f:
                    key, val = line.strip().split(': ')
                    result[key] = int(val)
            return result
        except:
            return {}

    def read_status(self, pid: int) -> dict:
        """
        /proc/<pid>/status — memory usage, thread count.
        """
        try:
            result = {}
            with open(f'/proc/{pid}/status') as f:
                for line in f:
                    if line.startswith('VmRSS'):
                        result['vm_rss_kb'] = int(line.split()[1])
                    elif line.startswith('Threads'):
                        result['threads'] = int(line.split()[1])
            return result
        except:
            return {}

# Default feature vector for cold start (unknown processes)
DEFAULT_FEATURES = np.array(
    [0.5, 0.1, 0.3, 0.5, 0.1, 0.2, 0.86, 0.1, 0.5, 1.0],
    dtype=np.float32
)

class FeatureExtractor:
    """
    Combines EventBuffer + ProcReader into a normalised feature vector.
    This is what the PPO agent receives as its 'state'.
    """

    # Normalisation constants — max expected values
    MAX_CPU_TIME_NS  = 100_000_000_000  # 100 seconds in ns
    MAX_WAIT_TIME_NS = 100_000_000_000
    MAX_TIMESLICES   = 1_000_000
    MAX_IO_BYTES     = 10_000_000_000   # 10GB
    MAX_THREADS      = 1000

    def __init__(self):
        self.proc   = ProcReader()
        self.buffer = EventBuffer(maxlen=50)

    def add_event(self, event: dict):
        """Feed eBPF events in — call this from your probe callback."""
        self.buffer.add_event(event)

    def extract(self, pid: int) -> np.ndarray:
        """
        Returns a 10-element float32 array for the given PID.
        All values normalised to [0.0, 1.0].
        """
        stat      = self.proc.read_stat(pid)
        schedstat = self.proc.read_schedstat(pid)
        io        = self.proc.read_io(pid)
        status    = self.proc.read_status(pid)

        # Cold start — process just appeared, no data yet
        if not stat and not schedstat:
            return DEFAULT_FEATURES.copy()

        # Feature 1: CPU usage ratio
        cpu_time = schedstat.get('cpu_time_ns', 0)
        f1 = min(cpu_time / self.MAX_CPU_TIME_NS, 1.0)

        # Feature 2: I/O ratio (read + write bytes)
        io_bytes = io.get('read_bytes', 0) + io.get('write_bytes', 0)
        f2 = min(io_bytes / self.MAX_IO_BYTES, 1.0)

        # Feature 3: avg wait time (starvation indicator)
        wait_time = schedstat.get('wait_time_ns', 0)
        f3 = min(wait_time / self.MAX_WAIT_TIME_NS, 1.0)

        # Feature 4: avg burst length (cpu_time / timeslices)
        slices = schedstat.get('timeslices', 1)
        burst  = cpu_time / max(slices, 1)
        f4 = min(burst / 10_000_000, 1.0)  # normalise to 1ms max burst

        # Feature 5: timeslice count (scheduling frequency)
        f5 = min(slices / self.MAX_TIMESLICES, 1.0)

        # Feature 6: memory pressure
        mem_kb = status.get('vm_rss_kb', 0)
        f6 = min(mem_kb / self.proc.total_ram_kb, 1.0)

        # Feature 7: priority level (120 = default, lower = higher priority)
        priority = stat.get('priority', 120)
        f7 = max(0.0, min(priority / 139.0, 1.0))  # normalise to max priority value

        # Feature 8: thread count
        threads = status.get('threads', 1)
        f8 = min(threads / self.MAX_THREADS, 1.0)

        # Feature 9: recent state pattern (CPU vs I/O bound)
        f9 = self.buffer.get_state_pattern(pid)

        # Feature 10: CPU affinity score (cache locality)
        f10 = self.buffer.get_cpu_affinity_score(pid)

        return np.array(
            [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10],
            dtype=np.float32
        )

if __name__ == "__main__":
    import sys

    extractor = FeatureExtractor()
    # Test on a real PID — pass it as argument
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else os.getpid()

    print(f"Extracting features for PID {pid}...")
    features = extractor.extract(pid)

    labels = [
        "cpu_usage", "io_ratio", "wait_time", "burst_len",
        "timeslices", "memory", "priority", "threads",
        "state_pattern", "cpu_affinity"
    ]

    print("\nFeature Vector:")
    for label, value in zip(labels, features):
        bar = " " * int(value * 30)
        print(f"  {label:>15}: {value:.4f}  {bar}")
