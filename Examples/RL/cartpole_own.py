from Controllers.RL.DDPGTrainer import DDPGTrainer

from Models.rl_model_own_cartpole import CustomContinuousCartPoleEnv


env = CustomContinuousCartPoleEnv()

ddpg_controller = DDPGTrainer(env)

ddpg_controller.run()





