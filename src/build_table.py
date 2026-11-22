"""
Assemble the paper's Table 1: probability that answers conform with each moral
value, for the PT (base) model vs each Moral Agent (diagonal = each agent on
its own value). Reads results/conformance/*.json written by eval_conformance.py.
"""
import json
import os

from common import VALUES

CONF = "results/conformance"


def load(label):
    path = os.path.join(CONF, f"{label}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))["means"]


def main():
    pt = load("PT")
    agents = {v: load(f"agent_{v}") for v in VALUES}

    print("\n=== Table 1: P(answer conforms to moral value) ===\n")
    header = f"{'Moral Value':12s} | {'PT model':>10s} | {'Moral Agent':>12s}"
    print(header)
    print("-" * len(header))
    rows = {}
    for v in VALUES:
        pt_v = pt[v] * 100 if pt else float("nan")
        ag_v = agents[v][v] * 100 if agents.get(v) else float("nan")
        rows[v] = {"PT": pt_v, "agent": ag_v}
        print(f"{v:12s} | {pt_v:9.2f}% | {ag_v:11.2f}%")

    # Full cross-table (each agent scored on all values) if available
    print("\n=== Full agent x value conformance (%) ===")
    print(f"{'agent\\\\value':14s} " + " ".join(f"{v[:6]:>7s}" for v in VALUES))
    for a in VALUES:
        m = agents.get(a)
        if not m:
            continue
        print(f"{a:14s} " + " ".join(f"{m[v]*100:7.1f}" for v in VALUES))

    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/table1.json", "w"), indent=2)
    print("\nSaved results/table1.json")


if __name__ == "__main__":
    main()
