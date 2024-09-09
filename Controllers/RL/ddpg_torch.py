"""
For reference this code is based on the following repository:
https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master/DDPG
"""

import os
import numpy as np
import torch as T
import torch.nn.functional as F
from Controllers.RL.networks import ActorNetwork, CriticNetwork
from Controllers.RL.noise import OUActionNoise
from Controllers.RL.buffer import ReplayBuffer

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, model_name, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300, 
                 batch_size=100, action_bound=2, sigmoid=False, sigma_noise=0.5, chkpt_dir='tmp/ddpg', dt=0.1, theta_noise=0.15):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.action_bound = action_bound

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions), sigma=sigma_noise, dt=dt, theta=theta_noise)

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor', model_name=model_name, action_bound=self.action_bound, chkpt_dir=chkpt_dir, sigmoid=sigmoid)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic', model_name=model_name, chkpt_dir=chkpt_dir)
        
        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor', model_name=model_name, action_bound=self.action_bound, chkpt_dir=chkpt_dir, sigmoid=sigmoid)
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic', model_name=model_name, chkpt_dir=chkpt_dir)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        
        if not observation.shape:
            observation = np.array([observation])

        state = T.tensor([observation], dtype=T.float).to(self.actor.device)


        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)
        
        action_bound = T.tensor(self.action_bound).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, -action_bound, action_bound)
        self.actor.train()

        action = mu_prime.cpu().detach().numpy()[0]

        return action
    
    def choose_action_eval(self, observation):
        self.actor.eval()
        
        if not observation.shape:
            observation = np.array([observation])

        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        # Directly use the actor network without adding noise
        mu = self.actor.forward(state).to(self.actor.device)
        action = mu.cpu().detach().numpy()[0]

        return action

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)
