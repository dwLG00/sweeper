import gym
import gym.spaces
import torch
from torch.distributions import Categorical
import numpy as np
from .minesweeper import Board

device = torch.device('cpu')

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class MinesweeperActor(torch.nn.Module):
    def __init__(self, w, h):
        super().__init__()
        self.actor_mlp = torch.nn.Sequential(
            torch.nn.Linear((w - 2) * (h - 2), 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 50),
            torch.nn.ReLU()
        )
        self.x_net = torch.nn.Sequential(
            torch.nn.Linear(50, w),
        )
        self.y_net = torch.nn.Sequential(
            torch.nn.Linear(50, h),
        )
        self.action_net = torch.nn.Sequential(
            torch.nn.Linear(50, 2),
        )

    def forward(self, conv_out):
        x_dist, y_dist, a_dist = self.dists(conv_out)
        x, y, a = x_dist.sample(), y_dist.sample(), a_dist.sample()
        log_probs = x_dist.log_prob(x).detach() + y_dist.log_prob(y).detach() + a_dist.log_prob(a).detach()
        concatted_action = torch.cat((x.detach(), y.detach(), a.detach()))
        return concatted_action, log_probs
    
    def dists(self, conv_out):
        v = self.actor_mlp(conv_out)
        x_logits = self.x_net(v)
        y_logits = self.y_net(v)
        a_logits = self.action_net(v)
        assert torch.isfinite(x_logits).all()
        assert torch.isfinite(y_logits).all()
        assert torch.isfinite(a_logits).all()
        x_dist, y_dist, a_dist = Categorical(logits=x_logits), Categorical(logits=y_logits), Categorical(logits=a_logits)
        return x_dist, y_dist, a_dist



class MinesweeperActorCritic(torch.nn.Module):
    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h
        self.conv_layer = torch.nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3, stride=1)

        # actor
        self.actor = MinesweeperActor(w, h)

        # critic
        self.critic = torch.nn.Sequential(
            torch.nn.Linear((w - 2) * (h - 2), 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1)
        )
    
    def act(self, state):
        v = self.conv_layer(state)
        v = v.view(v.size(0), -1)
        concatted_action, log_probs = self.actor(v)
        state_val = self.critic(v)
        return concatted_action, log_probs, state_val.detach()
    
    def evaluate(self, state, action):
        v = self.conv_layer(state)
        v = v.view(v.size(0), -1)
        x_dist, y_dist, a_dist = self.actor.dists(v)
        x, y, a = action[:, 0], action[:, 1], action[:, 2]

        action_logprobs = x_dist.log_prob(x) + y_dist.log_prob(y) + a_dist.log_prob(a)
        entropy = x_dist.entropy() + y_dist.entropy() + a_dist.entropy()
        state_values = self.critic(v)

        return action_logprobs, state_values, entropy

class PPO:
    def __init__(self, w, h, lr_actor, lr_critic, lr_conv, gamma, K_epochs, clip):
        self.w = w
        self.h = h
        self.policy = MinesweeperActorCritic(w, h).to(device)
        self.buffer = RolloutBuffer()

        self.gamma = gamma
        self.K_epochs = K_epochs
        self.clip = clip

        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.conv_layer.parameters(), 'lr': lr_conv},
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        self.policy_old = MinesweeperActorCritic(w, h).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.loss = torch.nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action
    
    def update(self):
        rewards = []
        discounted_reward = 0
        # compute rewards
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward) # r'_{t - 1} = r_{t - 1} + \gamma * r'_t
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        advantages = rewards.detach() - old_state_values.detach()

        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))