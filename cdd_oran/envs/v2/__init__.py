"""v2 benchmark environments (SEMANTICS.md-conformant code path).

This subpackage is a NEW code path, deliberately independent of the legacy
``cdd_oran.envs.legacy.base.BaseORANEnv`` and ``env_i..iv`` modules so that legacy behavior stays
byte-identical (benchmark-version boundary, PD4 / review MAJOR 7). It implements the
apply/advance split, one-step actuation latency, a **coordinate-keyed exogenous tape** whose
draws are a pure function of ``(env_seed, episode, time, variable)`` (never a mutable
sequential stream), a real noise-off knob, and a latent-noiseless outcome accessor distinct
from the observed (noisy) outcome (SEMANTICS §1.1/§2/§4/§5).
"""

from cdd_oran.envs.v2.base import V2Env
from cdd_oran.envs.v2.e1 import E1V2Env
from cdd_oran.envs.v2.e2 import E2V2Env
from cdd_oran.envs.v2.e3 import E3V2Env
from cdd_oran.envs.v2.e4 import E4V2Env

__all__ = ["V2Env", "E1V2Env", "E2V2Env", "E3V2Env", "E4V2Env"]
