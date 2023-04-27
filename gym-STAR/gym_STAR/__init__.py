from gymnasium.envs.registration import register

register(
    id='gym_STAR/My_Env-v1',
    entry_point='gym_STAR.env:My_Env',
    nondeterministic=True
)

register(
    id='gym_STAR/RIS_Env-v1',
    entry_point='gym_STAR.env:RIS_Env',
    nondeterministic=True
)

register(
    id='gym_STAR/Fix_Pos-v1',
    entry_point='gym_STAR.env:Fix_Pos',
    nondeterministic=True
)

register(
    id='gym_STAR/Fix-v1',
    entry_point='gym_STAR.env:Fix',
    nondeterministic=True
)