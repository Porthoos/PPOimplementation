from gym.envs.registration import register

register(
    id='gym_STAR/My_Env-v1',
    entry_point='gym_STAR.env:My_Env',
    nondeterministic=True
)