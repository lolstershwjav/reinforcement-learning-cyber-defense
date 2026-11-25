"""Baselines over the same simulated observation and action space."""

import numpy as np

from defense_sim.environment import N


def rule(name,observation,env):
    if name=="wait":
        return 0
    obs=observation[:-1].reshape(N,6)
    alert,isolated,confirmed,clean,recovery,critical=obs.T
    if isolated.any():
        return int(1+2*N+np.argmax(isolated))
    if name=="oracle":
        danger=(env.active>0)&(recovery==0)
        return int(1+N+np.argmax(danger*critical)) if danger.any() else 0
    if name=="inspect_rule" and confirmed.any():
        return int(1+N+np.argmax(confirmed*critical))
    score=alert*(.5+.5*critical)*(recovery==0)*(clean==0)
    i=int(np.argmax(score))
    if name=="alert_rule":
        return 1+N+i if score[i]>.61 else 0
    if name=="inspect_rule":
        return 1+i if score[i]>.44 else 0
    raise ValueError(f"Unknown policy {name!r}")
