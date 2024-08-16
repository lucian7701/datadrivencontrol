from Controllers.RL.DDPGTrainer import DDPGTrainer
from Controllers.RL.DDPGEvaluator import DDPGEvaluator
from Models.rl_model_own_di import CustomDoubleIntegratorEnv

env = CustomDoubleIntegratorEnv(target_position=7.0)

ddpg_controller = DDPGTrainer(env, input_dims=[1], n_actions=1)

ddpg_controller.run()

ddpg_eval_controller = DDPGEvaluator(env, input_dims=[1], n_actions=1)

ddpg_eval_controller.run_eval()

