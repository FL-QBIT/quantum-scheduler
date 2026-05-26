# reward.py
import os

class RewardComputer:
    """
    Computes reward signal from real system metrics.
    Reads directly from /proc — no simulation.
    """

    def __init__(self, target_pids: list):
        self.target_pids = target_pids
        self.prev_cpu_times = {}  # for delta computation
        self.prev_wait_times = {}

    def compute(self) -> float:
        """
        Returns a single reward float for the current system state.
        Positive = good scheduling, negative = bad scheduling.
        """
        throughput_score = self._throughput()
        fairness_score   = self._fairness()
        wait_penalty     = self._wait_penalty()
        util_score       = self._cpu_utilisation()

        reward = (
            + 0.30 * throughput_score
            + 0.25 * util_score
            + 0.20 * fairness_score
            - 0.15 * wait_penalty
            - 0.10 * self._latency_proxy()
        )

        return float(reward)

    def _throughput(self) -> float:
        """
        Measures CPU progress across tracked PIDs.
        Higher delta cpu_time = more work done = higher throughput.
        """
        total_delta = 0
        for pid in self.target_pids:
            try:
                with open(f'/proc/{pid}/stat') as f:
                    fields = f.read().split()
                cpu_time = int(fields[13]) + int(fields[14])
                prev     = self.prev_cpu_times.get(pid, cpu_time)
                total_delta += cpu_time - prev
                self.prev_cpu_times[pid] = cpu_time
            except:
                continue

        # Normalise — 1000 ticks delta = score of 1.0
        return min(total_delta / 1000.0, 1.0)    

    # Replace _fairness() with:
    def _fairness(self) -> float:
        delta_waits = []
        for pid in self.target_pids:
            try:
                with open(f'/proc/{pid}/schedstat') as f:
                    _, wait_time, _ = map(int, f.read().split())
                prev  = self.prev_wait_times.get(pid, wait_time)
                delta = wait_time - prev
                self.prev_wait_times[pid] = wait_time
                delta_waits.append(delta)
            except:
                continue

        if len(delta_waits) < 2:
            return 1.0

        n    = len(delta_waits)
        s    = sum(delta_waits)
        sq_s = sum(w ** 2 for w in delta_waits)
        return (s ** 2) / (n * sq_s + 1e-8)

    # Replace _wait_penalty() with:
    def _wait_penalty(self) -> float:
        delta_waits = []
        for pid in self.target_pids:
            try:
                with open(f'/proc/{pid}/schedstat') as f:
                    _, wait_time, _ = map(int, f.read().split())
                prev  = self.prev_wait_times.get(pid, wait_time)
                delta_waits.append(wait_time - prev)
            except:
                continue

        if not delta_waits:
            return 0.0

        avg_wait = sum(delta_waits) / len(delta_waits)
        return min(avg_wait / 500_000_000, 1.0)  # 500ms = max penalty

    def _cpu_utilisation(self) -> float:
        """
        System-wide CPU utilisation from /proc/stat.
        Higher = less waste.
        """
        try:
            with open('/proc/stat') as f:
                line = f.readline().split()
            user    = int(line[1])
            nice    = int(line[2])
            system  = int(line[3])
            idle    = int(line[4])
            total   = user + nice + system + idle
            active  = user + nice + system
            return active / max(total, 1)
        except:
            return 0.5

    def _latency_proxy(self) -> float:
        """
        Uses avg wait_time as a proxy for interactive latency.
        Real latency measurement requires kernel instrumentation.
        """
        return self._wait_penalty()

#TEST
if __name__ == "__main__":
    import time
    import subprocess

    # Get ALL active PIDs not just firefox
    # More PIDs = more CPU activity = reward actually changes
    result = subprocess.run(
        ['ps', '--no-headers', '-eo', 'pid', '--sort=-%cpu'],
        capture_output=True, text=True
    )
    # Take top 20 most active processes
    pids = []
    for line in result.stdout.strip().split('\n')[:20]:
        try:
            pids.append(int(line.strip()))
        except:
            continue

    print(f"Tracking {len(pids)} PIDs: {pids[:5]}...")
    computer = RewardComputer(pids)

    # First call establishes baseline
    computer.compute()

    # Now sample — delta will be real
    for i in range(10):
        time.sleep(0.5)
        r = computer.compute()

        # Break down reward components for visibility
        t = computer._throughput()
        f = computer._fairness()
        w = computer._wait_penalty()
        u = computer._cpu_utilisation()

        print(f"Step {i+1:2d}: reward={r:+.4f}  "
              f"throughput={t:.3f}  "
              f"fairness={f:.3f}  "
              f"wait={w:.3f}  "
              f"util={u:.3f}")