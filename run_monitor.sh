#!/bin/zsh
# launchd entry for the live monitor (com.mkampo.ragpoliciesmonitor).
#
# HOURLY, not every few minutes, and the reason is the probe: it asks a real
# question, so a 10-minute cadence would be ~144 questions a day and would
# exhaust the free tier on its own - monitoring that causes the outage it
# watches for. Hourly is ~24/day against a ~75-question budget, which is
# affordable and still catches an outage within the hour.
#
# Everything except the probe is free (it reads files), so if the probe ever
# becomes too expensive, --no-probe keeps eight of the ten detectors running.
set -e
cd "$(dirname "$0")"
[ -f "$HOME/.config/ragpolicies/env" ] && source "$HOME/.config/ragpolicies/env"
exec .venv/bin/python3 run_monitor.py "$@"
