"""Flow Reversal Steering (FRS, arXiv:2606.13675) experiments.

- `noise_policy`: DSBC noise policy (small MLP trained by BC on recovered FRS noise).
- `policy_bridge`: load a flow-matching policy checkpoint and call its FRS paths.
- `reference_builder`: keyboard direction -> IK -> reference action chunk (SO-101).
"""
