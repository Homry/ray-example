from pyglet.window import key
import ray
from ray import tune
from ray.rllib.agents.ddpg import ddpg
from ray.tune.registry import register_env
from env import launch_env

if __name__ == "__main__":
    ray.init()
    config = ddpg.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 2
    config["env"] = 'Duckietown'
    register_env('Duckietown', launch_env)
    tune.run(
        "DDPG",
        stop={"episode_reward_mean": 200},
        config={
            "env": "Duckietown",
            "num_gpus": 2,
            "num_workers": 1,
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        },
    )
