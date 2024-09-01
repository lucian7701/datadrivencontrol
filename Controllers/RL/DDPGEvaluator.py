import numpy as np
import matplotlib.pyplot as plt
from Controllers.RL.ddpg_torch import Agent
from Analysis.state_control_reference import plot_results
from Analysis.Analyser import Analyser

class DDPGEvaluator:

    def __init__(self, env, model_name, input_dims=[4], n_actions=1, total_simulation_time=20, dt=0.1, action_bound=2, sigmoid=False, problem='tmp/ddpg'):
        self.env = env
        self.problem = problem
        self.chkpt_dir = f"trained_agents/{problem}"
        self.agent = Agent(alpha=0.0001, beta=0.001, 
                           input_dims=input_dims, tau=0.005,
                           batch_size=64, fc1_dims=400, fc2_dims=300, 
                           n_actions=n_actions, model_name=model_name, action_bound=action_bound, sigmoid=sigmoid, chkpt_dir=self.chkpt_dir)
        self.agent.load_models()  # Load trained models
        self.dt = dt
        self.total_simulation_time = total_simulation_time
        self.max_steps_per_episode = total_simulation_time // dt
        self.u = None
        self.y = None

    def run_eval(self, state_labels=None, control_labels=None, ref_labels=None, plot: bool = True):
        observation, _ = self.env.reset()
        done = False

        rewards = []

        step_num = 0

        while not done and step_num<self.max_steps_per_episode:

            # check what type observation is
            self.y = np.vstack((self.y, observation)) if self.y is not None else observation
            
            action = self.agent.choose_action_eval(observation)

            # check what type action is
            self.u = np.vstack((self.u, action)) if self.u is not None else action

            if action.shape[0] == 1:
                action = action.item()
            observation_, reward, done, info, _ = self.env.step(action, eval=True)
            rewards.append(reward)

            observation = observation_

            step_num += 1

        analyser = Analyser(states=self.y, controls=self.u, reference_trajectory=np.array(self.env.target_positions).reshape(1,-1))
        
        total_absolute_error_by_state = analyser.total_absolute_error_by_state()
        total_absolute_control_by_input = analyser.total_absolute_control_by_input()
     

        print("Total absolute error:", analyser.total_absolute_error())
        
        if plot:
            analyser.plot_state_control(self.dt, state_labels, control_labels, ref_labels)
        

        print("Number of steps:", step_num)
        rewards = np.array(rewards)

        return total_absolute_error_by_state, total_absolute_control_by_input
