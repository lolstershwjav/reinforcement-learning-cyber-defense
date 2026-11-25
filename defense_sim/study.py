"""Learn defensive policies and compare on paired simulated incident tapes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from stable_baselines3 import DQN, PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from defense_sim.environment import BALANCED, DefenseEnv
from defense_sim.policies import rule

RULES=("wait","alert_rule","inspect_rule","oracle")
ARMS={
    "dqn": ("dqn","balanced",True),
    "ppo": ("ppo","balanced",True),
    "ppo_security": ("ppo","security",True),
    "ppo_availability": ("ppo","availability",True),
    "ppo_no_inspection": ("ppo","balanced",False),
    "ppo_masked": ("masked","balanced",True),
    "ppo_masked_prior": ("masked_prior","balanced",True),
    "imitation_only": ("masked_bc","balanced",True),
}
SCENARIOS=("iid","bursty","shifted")
FIELDS=("exposure","outage","cost","invalid","collateral","resolved")


def rollout(policy, scenario, seeds, horizon, reward=BALANCED, inspection=True, trace=False):
    envs=[DefenseEnv(horizon,scenario,reward,inspection) for _ in seeds]
    observations=np.stack([e.reset(seed=seed)[0] for e,seed in zip(envs,seeds,strict=True)])
    values=np.zeros((len(envs),len(FIELDS)+1))
    evidence=[]
    for step in range(horizon):
        if isinstance(policy,str):
            action=[rule(policy,o,e) for o,e in zip(observations,envs,strict=True)]
        elif isinstance(policy,MaskablePPO):
            masks=np.stack([e.action_masks() for e in envs])
            action=policy.predict(observations,deterministic=True,action_masks=masks)[0]
        else:
            action=policy.predict(observations,deterministic=True)[0]
        future=[]
        for i,(cell,act) in enumerate(zip(envs,action,strict=True)):
            obs,rt,_,_,info=cell.step(act)
            future.append(obs)
            values[i]+=np.array([info[k] for k in FIELDS]+[rt])
            if trace and i==0:
                evidence.append(dict(step=step,reward=rt,**info))
        observations=np.stack(future)
    values[:,:3]/=horizon
    rows=[dict(episode_seed=int(seed),**dict(zip((*FIELDS,"reward"),map(float,v),strict=True)))
          for seed,v in zip(seeds,values,strict=True)]
    return rows,evidence


class Checkpoints(BaseCallback):
    def __init__(self, steps, horizon, weights, inspection):
        super().__init__()
        self.steps=set(steps)
        self.horizon=horizon
        self.weights=weights
        self.inspection=inspection
        self.history=[]
        self.best=-float("inf")
        self.state=None

    def _on_step(self):
        if self.num_timesteps not in self.steps:
            return True
        average=validation_score(self.model,self.horizon,self.weights,self.inspection)
        self.history.append(dict(steps=self.num_timesteps,mean_reward=average))
        if average>self.best:
            self.best=average
            self.state=deepcopy(self.model.policy.state_dict())
        return True


def validation_score(model,horizon,weights,inspection):
    rewards=[]
    for scenario in ("iid","bursty"):
        rows,_=rollout(model,scenario,[180,181,182,183],horizon,weights,inspection)
        rewards.extend(r["reward"] for r in rows)
    return float(np.mean(rewards))


def imitate_rule(model, seed, horizon):
    observations,actions,masks=[],[],[]
    env=DefenseEnv(horizon,"train")
    for episode in range(40):
        obs,_=env.reset(seed=10000+seed*100+episode)
        for _ in range(horizon):
            action=rule("inspect_rule",obs,env)
            observations.append(obs)
            actions.append(action)
            masks.append(env.action_masks())
            obs,*_=env.step(action)
    x=torch.tensor(np.stack(observations))
    a=torch.tensor(actions)
    allowed=np.stack(masks)
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=.002)
    rng=torch.Generator().manual_seed(seed)
    for _ in range(16):
        for idx in torch.randperm(len(x),generator=rng).split(128):
            dist=model.policy.get_distribution(x[idx],action_masks=allowed[idx.numpy()])
            loss=-dist.log_prob(a[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def learn(name, seed, cfg):
    algo,objective,inspection=ARMS[name]
    weights=cfg["rewards"][objective]
    horizon=cfg["horizon"]
    env=DummyVecEnv([lambda:DefenseEnv(horizon,"train",weights,inspection) for _ in range(4)])
    shared=dict(seed=seed,verbose=0,device="cpu",gamma=.97,policy_kwargs=dict(net_arch=[64,64]))
    if algo=="dqn":
        model=DQN("MlpPolicy",env,buffer_size=18000,learning_starts=600,
                  batch_size=96,train_freq=4,gradient_steps=2,target_update_interval=1000,
                  learning_rate=.0007,exploration_fraction=.42,exploration_final_eps=.06,**shared)
    else:
        algorithm=MaskablePPO if algo.startswith("masked") else PPO
        model=algorithm("MlpPolicy",env,n_steps=128,batch_size=128,n_epochs=6,
                  learning_rate=.0006,ent_coef=.013,**shared)
    checkpoint=Checkpoints(cfg["checkpoints"],horizon,weights,inspection)
    start=perf_counter()
    if algo in {"masked_prior","masked_bc"}:
        imitate_rule(model,seed,horizon)
    if algo=="masked_bc":
        history=[dict(steps=0,mean_reward=validation_score(model,horizon,weights,inspection))]
        env.close()
        return model,history,perf_counter()-start
    model.learn(total_timesteps=cfg["checkpoints"][-1],callback=checkpoint)
    seconds=perf_counter()-start
    if checkpoint.state is None:
        raise RuntimeError("No checkpoint selected")
    model.policy.load_state_dict(checkpoint.state)
    env.close()
    return model,checkpoint.history,seconds


def paired(rows, seed=538):
    rng=np.random.default_rng(seed)
    result=[]
    for scenario in SCENARIOS:
        arm=[r for r in rows if r["scenario"]==scenario]
        baseline={r["episode_seed"]:r for r in arm if r["model"]=="alert_rule"}
        for name in (*RULES,*ARMS):
            model=[r for r in arm if r["model"]==name]
            runs=sorted({r["train_seed"] for r in model})
            eval_seeds=sorted(baseline)
            out={}
            for field in (*FIELDS,"reward"):
                matrix=np.array([[next(r[field] for r in model if
                                       r["train_seed"]==train and r["episode_seed"]==ep)
                                  for ep in eval_seeds] for train in runs])
                # Resample scenario seeds across policies after averaging training repeats.
                means=matrix.mean(0)
                indices=rng.integers(0,len(means),(4000,len(means)))
                for metric,vals in ((field,means),(field+"_vs_rule",
                            means-np.asarray([baseline[ep][field] for ep in eval_seeds]))):
                    lower,upper=np.quantile(vals[indices].mean(1),[.025,.975])
                    out[metric]=dict(mean=float(vals.mean()),low=float(lower),high=float(upper))
            result.append(dict(scenario=scenario,model=name,metrics=out))
    return result


def run(cfg):
    torch.set_num_threads(1)
    seeds=list(range(cfg["test_seed_start"],cfg["test_seed_start"]+cfg["test_episodes_per_scenario"]))
    if set(seeds)&{180,181,182,183}:
        raise ValueError("Validation and test scenario seeds must be disjoint")
    if not cfg["checkpoints"] or any(step%512 for step in cfg["checkpoints"]):
        raise ValueError("Checkpoints must align to 512-transition PPO rollout boundaries")
    results,training,demo=[],[],[]

    def testing(policy,name,seed,inspection):
        for scenario in SCENARIOS:
            rows,trace=rollout(policy,scenario,seeds,cfg["horizon"],inspection=inspection,trace=True)
            results.extend(dict(model=name,scenario=scenario,train_seed=seed,**row) for row in rows)
            if seed in (-1,cfg["train_seeds"][0]):
                demo.append(dict(model=name,scenario=scenario,episode_seed=seeds[0],timeline=trace))

    for name in RULES:
        testing(name,name,-1,True)
    for name in ARMS:
        for seed in cfg["train_seeds"]:
            model,history,seconds=learn(name,seed,cfg)
            training.append(dict(model=name,seed=seed,history=history,seconds=seconds))
            testing(model,name,seed,ARMS[name][2])
            print(f"Finished {name}, training seed {seed}",flush=True)
    return dict(config=cfg,rows=results,summary=paired(results),training=training,dashboard=demo)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path("configs/study.json"))
    p.add_argument("--output",type=Path,default=Path("results/study.json"))
    args=p.parse_args()
    result=run(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(result,separators=(",",":"),allow_nan=False))
    temp.replace(args.output)


if __name__=="__main__":
    main()
