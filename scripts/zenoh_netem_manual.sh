#!/usr/bin/env bash
# Real kernel-level network impairment for the Zenoh validation scenarios
# (Algovalidations/zenoh.png, "Latency injection" and "Packet loss" rows).
#
# tc qdisc needs root/CAP_NET_ADMIN, which the automated test environment
# doesn't have non-interactively -- tests/validation/test_zenoh_validation.py
# validates the same properties with an application-level delay/loss
# stand-in instead (clearly labeled there as weaker than this). Run this
# script yourself for the authentic kernel-level version.
#
# Usage:
#   sudo ./scripts/zenoh_netem_manual.sh delay   # 100ms one-way delay on loopback
#   sudo ./scripts/zenoh_netem_manual.sh loss    # 10% packet loss on loopback
#   sudo ./scripts/zenoh_netem_manual.sh clear   # remove impairment
#
# While impairment is active, run the multi-process harness in another
# terminal to watch the fleet cope with it for real:
#   python3 -c "
#   from tests.validation.zenoh_worker import run_robot
#   import multiprocessing
#   procs = [multiprocessing.Process(target=run_robot, args=(f'r{i}', 'dispatcher' if i==0 else 'bidder', f'/tmp/r{i}.json', 15.0, (i*5.0,0.0))) for i in range(5)]
#   [p.start() for p in procs]; [p.join() for p in procs]
#   "
# then inspect /tmp/r*.json for pose_latencies_ms and winning_agents.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "This needs root (tc qdisc requires CAP_NET_ADMIN). Re-run with sudo." >&2
    exit 1
fi

case "${1:-}" in
    delay)
        tc qdisc replace dev lo root netem delay 100ms
        echo "Applied: 100ms delay on loopback (dev lo)."
        ;;
    loss)
        tc qdisc replace dev lo root netem loss 10%
        echo "Applied: 10% packet loss on loopback (dev lo)."
        ;;
    clear)
        tc qdisc del dev lo root 2>/dev/null || true
        echo "Cleared loopback impairment."
        ;;
    *)
        echo "Usage: sudo $0 {delay|loss|clear}" >&2
        exit 1
        ;;
esac

tc qdisc show dev lo
