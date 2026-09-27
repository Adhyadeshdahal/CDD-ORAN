"""E6 mandatory simple baselines (arbiters that use only ``obs``).

  no arbitration  : ``arbiter=None`` in ``E6Env.run`` (accept all, last writer wins)
  freeze          : reject every request (network stays at its initial 3GPP-default configuration)
  priority        : per knob and second, keep only the request of the highest-priority xApp; accept the rest
  knob_lock       : SON-coordination style lease: the first xApp to change a knob owns it for ``lease_s``; requests
                    by other xApps on an owned knob are rejected
"""
from __future__ import annotations

PRIORITY = ("SLICE", "MRO", "TS", "ES")


def freeze(obs):
    return {"decisions": ["reject"] * len(obs["requests"]), "writes": []}


def priority(obs, order=PRIORITY):
    rank = {x: i for i, x in enumerate(order)}
    best = {}
    for i, r in enumerate(obs["requests"]):
        k = r["knob"]
        if k not in best or rank.get(r["xapp"], 99) < rank.get(obs["requests"][best[k]]["xapp"], 99):
            best[k] = i
    keep = set(best.values())
    return {"decisions": ["accept" if i in keep else "reject" for i in range(len(obs["requests"]))], "writes": []}


class KnobLock:
    def __init__(self, lease_s=60.0):
        self.lease_s, self.owner = lease_s, {}

    def __call__(self, obs):
        now, dec = obs["t"], []
        for r in obs["requests"]:
            k = r["knob"]
            own = self.owner.get(k)
            if own is None or own[1] < now or own[0] == r["xapp"]:
                self.owner[k] = (r["xapp"], now + self.lease_s)
                dec.append("accept")
            else:
                dec.append("reject")
        return {"decisions": dec, "writes": []}


BASELINES = {"noarb": lambda: None, "freeze": lambda: freeze, "priority": lambda: priority, "lock": KnobLock}
