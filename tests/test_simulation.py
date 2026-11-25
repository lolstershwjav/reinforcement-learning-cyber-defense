"""Check exogenous scenarios, action invariants and masked policy training."""

import json
import unittest

from gymnasium.utils.env_checker import check_env
import numpy as np
import torch

from defense_sim.environment import BALANCED, DefenseEnv, N, tape
from defense_sim.study import learn, rollout


class DefenseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_gym_contract(self):
        check_env(DefenseEnv(), skip_render_check=True)

    def test_tape_is_seeded_and_shifted(self):
        same=tape(80,"iid",64)
        np.testing.assert_array_equal(same.ttl,tape(80,"iid",64).ttl)
        self.assertFalse(np.array_equal(same.ttl,tape(80,"bursty",64).ttl))
        self.assertFalse(np.array_equal(same.value,tape(80,"shifted",64).value))

    def test_isolation_and_recovery_affect_modeled_outcomes(self):
        e=DefenseEnv(scenario="iid")
        e.reset(seed=94)
        e.active[0]=7
        self.assertFalse(e.action_masks()[1+2*N])
        _,_,_,_, isolated=e.step(1+N)
        self.assertTrue(e.action_masks()[1+2*N])
        self.assertEqual(isolated["exposure"],0.)
        self.assertEqual(isolated["availability"][0],0.)
        self.assertLess(isolated["availability"][1],1.)
        _,rt,_,_,restored=e.step(1+2*N)
        self.assertEqual(restored["resolved"],1)
        self.assertAlmostEqual(rt,-np.dot(BALANCED,[restored[k] for k in ("exposure","outage","cost")]))

    def test_inspection_ablation_hides_confirmation(self):
        yes=DefenseEnv(inspection=True)
        no=DefenseEnv(inspection=False)
        for env in (yes,no):
            env.reset(seed=18)
            env.active[0]=7
        a,*_=yes.step(1)
        b,*_=no.step(1)
        self.assertEqual(a[2],1.)
        self.assertEqual(b[2],0.)

    def test_masked_training_and_evaluation(self):
        config={"horizon":32,"checkpoints":[512],
                "rewards":{"balanced":BALANCED}}
        model,log,_=learn("ppo_masked",43,config)
        self.assertEqual(log[0]["steps"],512)
        rows,trace=rollout(model,"shifted",[700,701],32,trace=True)
        self.assertEqual(len(trace),32)
        self.assertTrue(all(r["invalid"]==0 for r in rows))
        self.assertTrue(all(0<=r["outage"]<=1 for r in rows))
        json.dumps(rows,allow_nan=False)

    def test_rules_share_the_same_episode_ids(self):
        a,_=rollout("wait","iid",[70,71],32)
        b,_=rollout("oracle","iid",[70,71],32)
        self.assertEqual([r["episode_seed"] for r in a],[r["episode_seed"] for r in b])
        self.assertTrue(all(x["exposure"]>=y["exposure"] for x,y in zip(a,b,strict=True)))


if __name__=="__main__":
    unittest.main()
