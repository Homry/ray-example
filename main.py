import ray
from ray import tune
from ray.rllib.agents.ppo import ppo, PPOTrainer
from ray.rllib.agents.ddpg import ddpg_tf_model, ddpg, DDPGTrainer
from ray.tune.registry import register_env
import torch.optim as optim
from ray.tune.integration.torch import (DistributedTrainableCreator,
                                        distributed_checkpoint_dir)

from env import launch_env

if __name__ == "__main__":
    ray.init()
    register_env('Duckietown', launch_env)
    tune.run(
        DDPGTrainer,
        stop={"episode_reward_mean": 200},
        config={
            "env": "Duckietown",
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        },

    )

