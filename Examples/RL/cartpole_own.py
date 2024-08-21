from Controllers.RL.DDPGTrainer import DDPGTrainer
from Controllers.RL.DDPGEvaluator import DDPGEvaluator

from Models.rl_model_own_cartpole import CustomContinuousCartPoleEnv


env = CustomContinuousCartPoleEnv()

# ddpg_controller = DDPGTrainer(env, model_name="cartpole2", load_checkpoint=False)

# ddpg_controller.run()

ddpg_evaluator = DDPGEvaluator(env, model_name="cartpole")
ddpg_evaluator.run_eval()



# need to consider, what needs to go in the report from this: 
# We want to plot the reward against each episode as it trains. 

