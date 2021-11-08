import gym
from gym_duckietown.simulator import Simulator
from wrappers import *
from rewardWrapper import DtRewardWrapperDistanceTravelled

def launch_env():
    env = Simulator(
        seed=5123123,  # random seed
        map_name="loop_empty",
        max_steps=5000,  # we don't want the gym to reset itself
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=40,  # start close to straight
        full_transparency=True,
        distortion=True,
        domain_rand=False
    )
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapperDistanceTravelled(env)

    return env
