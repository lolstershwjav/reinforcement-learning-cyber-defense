"""Defensive incident response in a closed simulation with exogenous event tapes."""

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np

SERVICES = ("Gateway", "Identity", "Data store", "API", "Worker", "Monitoring")
N = len(SERVICES)
# Parent services affect availability if they are unavailable.
DEPENDENCIES = ((), (0,), (0,), (0,1,2), (0,2), (0,))
ACTIONS = ("Wait", "Inspect", "Isolate", "Restore")
BALANCED = (2.4,1.,.2)


@dataclass(frozen=True)
class Tape:
    ttl: np.ndarray
    telemetry: np.ndarray
    bump: np.ndarray
    value: np.ndarray


def tape(seed, scenario, horizon):
    rng = np.random.default_rng(seed)
    if scenario == "train":
        scenario = rng.choice(["iid","bursty"],p=[.72,.28])
    if scenario not in {"iid","bursty","shifted"}:
        raise ValueError(f"Unknown scenario {scenario!r}")
    rate = np.array([.009,.012,.019,.015,.021,.017])
    if scenario == "shifted":
        rate = np.array([.032,.025,.011,.021,.013,.024])
    entries = rng.random((horizon+1,N)) < rate
    if scenario == "bursty":
        at = rng.integers(8,horizon-12)
        cells = rng.choice(N,3,replace=False)
        entries[at+np.arange(3),cells] = True
    entries[rng.integers(1,10),rng.integers(N)] = True
    life = rng.integers(7,19,size=entries.shape)*entries
    bias = -.04 if scenario == "shifted" else 0.
    telemetry = rng.normal(bias,.19 if scenario=="shifted" else .13, size=entries.shape)
    bump = (rng.random(entries.shape) < (.13 if scenario=="shifted" else .06))*.41
    criticality = np.array([1.5,1.3,1.3,1.1,.8,.6])
    if scenario=="shifted":
        criticality = criticality[[3,2,1,0,5,4]]
    return Tape(life,telemetry,bump,criticality)


class DefenseEnv(gym.Env):
    """Actions affect only synthetic service state and scored outcomes."""

    metadata = {"render_modes":[]}

    def __init__(self,horizon=64,scenario="train",reward=BALANCED,inspection=True):
        super().__init__()
        if horizon < 24:
            raise ValueError("Episode length must be at least 24")
        self.horizon = horizon
        self.scenario = scenario
        self.reward_weights = reward
        self.inspection = inspection
        self.action_space = spaces.Discrete(1+3*N)
        self.observation_space = spaces.Box(0.,1.,shape=(N*6+1,),dtype=np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        effective = int(self.np_random.integers(1<<31)) if seed is None else int(seed)
        self.tape = tape(effective,self.scenario,self.horizon)
        self.step_id = 0
        self.active = self.tape.ttl[0].copy()
        self.isolated = np.zeros(N,dtype=bool)
        self.known = np.zeros(N,dtype=np.int8)
        self.freshness = np.zeros(N,dtype=np.int8)
        self.recovery = np.zeros(N,dtype=np.int8)
        return self.observe(), {}

    def observe(self):
        alert = np.clip(.14+.58*(self.active>0)+self.tape.telemetry[self.step_id]
                        +self.tape.bump[self.step_id],0,1)
        inspected = (self.freshness>0) & self.inspection
        indicators = np.column_stack([alert,self.isolated,
                                     inspected & (self.known==1),
                                     inspected & (self.known==-1),
                                     self.recovery>0,self.tape.value/1.5])
        timeline = np.asarray([self.step_id/self.horizon])
        return np.concatenate([indicators.ravel(),timeline]).astype(np.float32)

    def action_masks(self):
        allowed = np.ones(self.action_space.n,dtype=bool)
        allowed[1+N:1+2*N] = ~self.isolated & (self.recovery==0)
        allowed[1+2*N:] = self.isolated
        return allowed

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Action outside simulated defense space")
        if self.step_id>=self.horizon:
            raise RuntimeError("Reset the scenario after an episode ends")
        verb, cell = (divmod(int(action)-1,N) if action else (-1,-1))
        cost, invalid, unnecessary, resolved = 0., False, False, False
        if verb==0:
            self.known[cell] = 1 if self.active[cell]>0 else -1
            self.freshness[cell] = 5
            cost=.09
        elif verb==1:
            invalid=bool(self.isolated[cell] or self.recovery[cell])
            if not invalid:
                self.isolated[cell] = True
                unnecessary = bool(self.active[cell]==0)
            cost=.22
        elif verb==2:
            invalid=not bool(self.isolated[cell])
            if not invalid:
                resolved = bool(self.active[cell]>0)
                self.active[cell] = 0
                self.isolated[cell] = False
                self.recovery[cell] = 1
                self.freshness[cell] = 0
            cost=.47
        if invalid:
            cost+=.18
        down = self.isolated | (self.recovery>0)
        availability = (~down).astype(float)
        for i, parents in enumerate(DEPENDENCIES):
            if parents and down[list(parents)].any():
                availability[i] *= .58
        affected=(self.active>0)&~self.isolated
        weight=self.tape.value
        security=float(weight@affected/weight.sum())
        outage=float(weight@(1-availability)/weight.sum())
        penalty=float(np.dot(self.reward_weights,[security,outage,cost]))
        metrics=dict(exposure=security,outage=outage,cost=cost,invalid=int(invalid),
                     collateral=int(unnecessary), resolved=int(resolved),
                     compromise_count=int((self.active>0).sum()))
        state=dict(active=self.active.astype(bool).astype(int).tolist(),
                   isolated=self.isolated.astype(int).tolist(),
                   availability=availability.tolist(),
                   action=ACTIONS[verb+1], service=SERVICES[cell] if action else None,
                   **metrics)
        self.step_id += 1
        self.active=np.maximum(self.active-1,self.tape.ttl[self.step_id])
        self.recovery=np.maximum(self.recovery-1,0)
        self.freshness=np.maximum(self.freshness-1,0)
        return self.observe(),-penalty,self.step_id>=self.horizon,False,state
