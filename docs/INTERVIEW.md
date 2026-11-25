# Technical interview preparation

1. **Why use a simulated enterprise?** It provides repeatable incident schedules and lets the study measure costs and availability without modifying any real system.
2. **What is partial observability here?** Policies see noisy alerts, service state and short-lived inspection evidence. The exact incident status is latent unless inspected.
3. **How do paired tapes improve evaluation?** Every policy starts with the same benchmark incident schedule, telemetry noise and asset values. Paired comparisons reduce variability from different scenarios.
4. **What is the reward trade-off?** Isolation lowers exposure, but it removes capacity and can degrade dependent services. Repeated inspections and repairs add costs.
5. **Why include two hand-written policies?** Alert and inspection rules provide strong transparent baselines. A wait-only comparison would hide whether learning was competitive.
6. **Why did the unmasked policies fail?** They could spend many actions requesting moves that were invalid in the current state. The rules and masks restrict decisions to state-compatible actions.
7. **How is the action mask constructed?** It checks observed isolation and recovery state. An isolate cannot target a service already isolated or recovering. A restore requires isolation.
8. **How is checkpoint selection kept separate?** Validation uses fixed independent tapes from the training scenario types. The final seeds and shifted scenario are excluded from training and selection.
9. **Does rule initialization prove the RL method improves on imitation?** The extra demonstrations supply prior behavior. Compare the imitation-only arm with its fine-tuned counterpart to assess what additional RL changes.
10. **What are the most important next checks?** Test alternate dependency rules, larger service inventories and calibrated telemetry from incident response specialists. Establish operational action and service constraints before transferring any policy.

## Evidence-based résumé drafts

- Built a six-service incident-response simulator and evaluated 3,360 paired policy-episode outcomes across independent, burst and shifted conditions.
- Reduced invalid decisions from 43.24 to zero per shifted episode by applying an observed-state feasibility mask to PPO.
- Isolated the contribution of imitation and reinforcement learning. The imitation-only policy achieved −13.49 balanced return versus −14.01 after fine-tuning on the shifted cohort.

The benchmark illustrates experimental design, careful reward accounting and a negative result worth discussing in a CrowdStrike research interview.
