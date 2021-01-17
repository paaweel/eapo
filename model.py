import gym
import gym_pepper
import torch as th
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import HER, SAC

class Model:
    def __init__(self, paramters = {}):
        """
        docstring
        """

        self.paramters = paramters
        self.env = TimeLimit(gym.make('PepperPush-v0'), max_episode_steps=100)

        self.model = HER(paramters.get("policy",'MlpPolicy'),
            self.env,
            SAC,
            online_sampling=paramters.get("online_sampling", False),
            verbose=paramters.get("verbose", 1),
            max_episode_length=paramters.get("max_episode_length", 100),
            buffer_size=paramters.get("buffer_size", 1000000),
            batch_size=paramters.get("batch_size", 256),
            learning_rate=paramters.get("learning_rate", 0.001),
            learning_starts=paramters.get("learning_starts", 1000),
            n_sampled_goal=paramters.get("n_sampled_goal", 4),
            gamma=paramters.get("gamma", 0.95),
            goal_selection_strategy=paramters.get("goal_selection_strategy", 'future'),
            ent_coef=paramters.get("ent_coef", 'auto'),
            policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 256, 256]),
            train_freq=paramters.get("train_freq", 1),
            tensorboard_log=paramters.get("tensorboard_log", "./data/0_tensorboard/")
        )
    

    def learn(self, iterations: int):
        self.model.learn(iterations)


    def save(self, path="./data/0"):
        self.model.save()


if __name__ == "__main__":
    model = Model()
    model.learn(100)
    
    
