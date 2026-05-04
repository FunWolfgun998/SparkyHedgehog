import stable_retro as retro
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from wrappers import SparkyDiscretizer, PreprocessFrame, SparkyReward, RewardScaler
import config

def make_env(game, state, env_index=0):
    def _init():
        env = retro.make(game=game, state=state, render_mode="rgb_array")

        env = SparkyDiscretizer(env)
        env = SparkyReward(env)
        env = RewardScaler(env)
        env = PreprocessFrame(env, size=config.IMG_SIZE)
        env = Monitor(env)

        return env
    return _init

def create_parallel_envs():
    env_fns =[make_env(config.GAME_NAME, config.STATE_NAME, env_index=i) for i in range(config.NUM_ENVS)]
    envs = SubprocVecEnv(env_fns)
    envs = VecFrameStack(envs, n_stack=config.FRAME_STACK)
    return envs