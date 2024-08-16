import gymnasium as gym
from Controllers.RL.DDPGTrainer import DDPGTrainer

from Models.rl_model_gym import ContinuousCartPoleEnv



env = gym.make('ContinuousCartPoleEnv')

ddpg_controller = DDPGTrainer(env)

ddpg_controller.run()

