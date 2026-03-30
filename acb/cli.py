"""
ACB command-line interface.

Usage:
    acb cbi --n-deployed 8 --pass-at-1 0.72 --overhead 0.082
    acb nstar --pass-at-1 0.72 --overhead 0.082
    acb pharm --n 5 --mu-a 0.72 --sigma-a 0.15 --mu-c 0.082
    acb crossover --c-supervisor 0.015 --c-all2all 0.082
"""

from __future__ import annotations

import argparse
import sys

from acb.cbi import interpret_cbi
from acb.model import optimal_fleet_size, p_harm, topology_crossover


def cmd_cbi(args):
    result = interpret_cbi(args.n_deployed, args.pass_at_1, args.overhead)
    print(result)


def cmd_nstar(args):
    n = optimal_fleet_size(args.pass_at_1, args.overhead)
    print(f"n* = {n}")
    print(f"  a = {args.pass_at_1}, c = {args.overhead}, a/c = {args.pass_at_1 / args.overhead:.2f}")


def cmd_pharm(args):
    p = p_harm(args.n, args.mu_a, args.sigma_a, args.mu_c)
    print(f"P(harm | n={args.n}) = {p:.4f}")
    print(f"  → {p*100:.1f}% chance that agent #{args.n + 1} degrades performance")


def cmd_crossover(args):
    n = topology_crossover(args.c_supervisor, args.c_all2all)
    print(f"n_crossover = {n}")
    print(f"  Supervisor routing outperforms all-to-all for all n >= {n}")


def main():
    parser = argparse.ArgumentParser(
        prog="acb",
        description="Agent Coordination Bound – CLI tools",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # CBI
    p_cbi = sub.add_parser("cbi", help="Compute Coordination Bottleneck Index")
    p_cbi.add_argument("--n-deployed", type=int, required=True, help="Current fleet size")
    p_cbi.add_argument("--pass-at-1", type=float, required=True, help="Per-agent accuracy (a)")
    p_cbi.add_argument("--overhead", type=float, required=True, help="Per-link overhead (c)")
    p_cbi.set_defaults(func=cmd_cbi)

    # n*
    p_ns = sub.add_parser("nstar", help="Compute optimal fleet size")
    p_ns.add_argument("--pass-at-1", type=float, required=True)
    p_ns.add_argument("--overhead", type=float, required=True)
    p_ns.set_defaults(func=cmd_nstar)

    # P(harm)
    p_ph = sub.add_parser("pharm", help="Compute degradation probability P(harm|n)")
    p_ph.add_argument("--n", type=int, required=True, help="Current fleet size")
    p_ph.add_argument("--mu-a", type=float, required=True, help="Mean accuracy gain")
    p_ph.add_argument("--sigma-a", type=float, required=True, help="Std dev of accuracy gain")
    p_ph.add_argument("--mu-c", type=float, required=True, help="Mean per-link overhead")
    p_ph.set_defaults(func=cmd_pharm)

    # Crossover
    p_cx = sub.add_parser("crossover", help="Compute topology crossover fleet size")
    p_cx.add_argument("--c-supervisor", type=float, required=True)
    p_cx.add_argument("--c-all2all", type=float, required=True)
    p_cx.set_defaults(func=cmd_crossover)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
