"""audit-classic2: ClassicBase.tune() ignores the arm (always default_config(): pc / granger -> 'eq')."""
from cdd_oran.xmethod.methods.classic import METHODS
from cdd_oran.xmethod.worlds.generate import generate_dataset
dev = [generate_dataset("E2", "R2", 300, s, kappa=0.25)[0] for s in (3_000_001, 3_000_002)]
pc = METHODS["pc"]()
cfg = pc.tune(dev)
nat = [pc.run(d, {"arm": "native"}) for d in dev]
from cdd_oran.xmethod.methods._classic_common import placebo_tau
print({"tune_cfg_arm": cfg["arm"], "tau_from_tune": cfg["tau"], "tau_native_runs": placebo_tau(nat)["tau"]})
try:
    pc.tune(dev, config={"arm": "native"})
except TypeError as e:
    print("no way to pass an arm:", e)
