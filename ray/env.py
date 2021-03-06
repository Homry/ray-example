try:
    import pyglet

    pyglet.window.Window()
except Exception:
    pass

import logging
import sys

try:
    from gym_duckietown.simulator import Simulator, DEFAULT_ROBOT_SPEED, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT
except Exception:
    pass
from wrappers.general_wrappers import InconvenientSpawnFixingWrapper
from wrappers.observe_wrappers import ResizeWrapper, NormalizeWrapper, ClipImageWrapper, MotionBlurWrapper, SegmentationWrapper, RandomFrameRepeatingWrapper, ObservationBufferWrapper, RGB2GrayscaleWrapper, LastPictureObsWrapper
from wrappers.reward_wrappers import DtRewardTargetOrientation, DtRewardVelocity, DtRewardCollisionAvoidance, DtRewardPosingLaneWrapper, DtRewardPosAngle
from wrappers.action_wpappers import Heading2WheelVelsWrapper, ActionSmoothingWrapper
from wrappers.envWrapper import ActionDelayWrapper, ForwardObstacleSpawnnigWrapper, ObstacleSpawningWrapper
from wrappers.aido_wrapper import AIDOWrapper
import random
import numpy as np

logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, env_config, seed):
        self._env_config = env_config
        self._env = None
        np.random.seed(seed)
        random.seed(seed)

    def create_env(self) -> Simulator:
        self._env = Simulator(
            seed=self._env_config["seed"],
            map_name=self._env_config["map_name"],
            max_steps=self._env_config["max_steps"],
            camera_width=self._env_config["camera_width"],
            camera_height=self._env_config["camera_height"],
            accept_start_angle_deg=self._env_config["accept_start_angle_deg"],
            full_transparency=self._env_config["full_transparency"],
            distortion=self._env_config["distortion"],
            domain_rand=self._env_config["domain_rand"],
        )
        self._wrap()
        return self._env

    def _wrap(self) -> None:
        #self._env = LastPictureObsWrapper(self._env)
        #self._env = ActionDelayWrapper(self._env)
        self._env = ClipImageWrapper(self._env, 3)
        #self._env = RGB2GrayscaleWrapper(self._env)
        self._env = ResizeWrapper(self._env, (84, 84))
        #self._env = RandomFrameRepeatingWrapper(self._env)
        #self._env = ObservationBufferWrapper(self._env)
        self._env = MotionBlurWrapper(self._env)
        self._env = NormalizeWrapper(self._env)
        self._env = Heading2WheelVelsWrapper(self._env)
        #self._env = ActionSmoothingWrapper(self._env)
        #self._env = DtRewardTargetOrientation(self._env)
        #self._env = DtRewardVelocity(self._env)
        #self._env = DtRewardCollisionAvoidance(self._env)
