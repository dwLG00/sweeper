from src.model import PPO
from src.env import MinesweeperGym
import itertools
import os
from pathlib import Path

def rollout(model: PPO, gym: MinesweeperGym):
    state = gym.reset()
    current_ep_reward = 0
    max_ep_len = model.w * model.h * 2
    success = False
    terminated = False
    for n_actions in range(max_ep_len):
        action = model.select_action(state)
        state, reward, done = gym.step(action)
        model.buffer.rewards.append(reward)
        model.buffer.is_terminals.append(done)
        current_ep_reward += reward
        if done:
            terminated = True
            break
    return current_ep_reward, n_actions, terminated

def train(log_dir, save_dir, train_for_epochs=-1):
    K_epochs = 100
    update_eps = 4
    log_every = 40
    clip = 0.2
    gamma = 0.99 # reward decay
    lr_actor = 0.0003
    lr_critic = 0.001
    lr_conv = 0.0001

    width, height, n_mines = 50, 20, 299
    ppo = PPO(width, height, lr_actor, lr_critic, lr_conv, gamma, K_epochs, clip)
    gym = MinesweeperGym(width, height, n_mines)

    log_dir = Path(log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    save_dir = Path(save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = lambda checkpoint_name: save_dir / ('PPO_%s.pth' % checkpoint_name)

    print('[*] Starting Training')
    print('[H] K_epochs: %s\t update_eps: %s\t clip: %s\t gamma: %s' % (K_epochs, update_eps, clip, gamma))
    print('[H] Learning Rates - actor: %s\t critic: %s\t convolution: %s' % (lr_actor, lr_critic, lr_conv))
    print('[M] Minesweeper board: w %s, h%s, %s mines' % (width, height, n_mines))
    print('[P] Saving to %s, logging to %s' % (save_dir, log_dir))

    iterator = itertools.count() if train_for_epochs < 0 else range(train_for_epochs)

    eps_rewards, eps_actions, eps_terminated = [], [], []
    for epoch in iterator:
        eps_reward, n_actions, terminated = rollout(ppo, gym)
        eps_rewards.append(eps_reward)
        eps_actions.append(n_actions)
        eps_terminated.append(terminated)

        if epoch % update_eps == 0:
            ppo.update()

        if epoch % log_every == log_every - 1:
            avg_reward = sum(eps_rewards) / log_every
            avg_actions = sum(eps_actions) / log_every
            rate_terminated = sum(1 if t else 0 for t in eps_terminated)
            rate_solved = sum(1 if eps_rewards[i] >= 0 and eps_terminated[i] else 0 for i in range(log_every))
            rate_failed = sum(1 if eps_rewards[i] < 0 and eps_terminated[i] else 0 for i in range(log_every))
            print()
            print('[*] Epoch %s' % epoch)
            print('[*] Avg Reward: %s' % avg_reward)
            print('[*] Avg # Actions: %s' % avg_actions)
            print('[*] Termination Rate: %s' % rate_terminated)
            print('[*] Solve Rate: %s' % rate_solved)
            print('[*] Fail Rate: %s' % rate_failed)
            eps_rewards, eps_actions, eps_terminated = [], [], []

if __name__ == '__main__':
    train('training/training_logs', 'training/saved_models')