#!/usr/bin/env python3
"""Keep a forge_guard reservation alive for as long as a PID lives.

The launchers load a ~18GB model into a long-lived server. forge_guard
(:8790) owns memory policy on this box and SIGTERMs any big process that
never went through admission control — which is exactly what killed the
Narrative Gemma server mid-request on 2026-08-03 (its CPU reads ~0 during a
GPU prefill, so evict-idle read it as an idle hog holding 18GB).

The shell lib acquires the lease synchronously *before* starting the server,
then hands the lease id and the server pid to this script, which heartbeats
until the server exits and then releases the seat.

    hold_mem_lease.py <lease_id> <server_pid> [ttl_seconds]

No guard on the box => nothing to do, exit quietly. A memory broker must
never be the reason a launcher can't start.
"""

import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/SongForgeM5"))

try:
    from mem_client import heartbeat, release
except Exception:
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        return 1
    lease_id = sys.argv[1]
    pid = int(sys.argv[2])
    ttl = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0
    if not lease_id:
        return 0

    try:
        while True:
            time.sleep(ttl / 3.0)
            try:
                os.kill(pid, 0)
            except OSError:
                break  # server gone — stop paying for its seat
            heartbeat(lease_id, ttl)
    finally:
        release(lease_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
