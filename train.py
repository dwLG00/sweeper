from src.model import PPO
from src.env import MinesweeperGym
from src.data import generate_pretraining_data
import itertools
import os
from pathlib import Path
import time
import torch
from tqdm import tqdm

def rollout(model: PPO, gym: MinesweeperGym, pretrain=False, mask=False):
    state = gym.reset()
    current_ep_reward = 0
    max_ep_len = model.w * model.h # any more and you are doing something really wrong
    terminated = False
    for n_actions in range(max_ep_len):
        action = model.select_action(state, mask=gym.mask() if mask else None)
        state, reward, done, won = gym.step(action) if not pretrain else gym.early_epoch_step(action)
        model.buffer.rewards.append(reward)
        model.buffer.is_terminals.append(done)
        current_ep_reward += reward
        if done:
            terminated = True
            break
    return current_ep_reward, n_actions, terminated, won

def pretrain(model: PPO, w, h, n, epochs=1, n_eps=10000):
    conv_layer = model.policy.conv_layer
    critic_layer = model.policy.critic

    print('[*] Generating pretraining dataset...')
    dataloader = generate_pretraining_data(w, h, n, n_eps=n_eps)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([
        {'params': conv_layer.parameters(), 'lr': 0.001},
        {'params': critic_layer.parameters(), 'lr': 0.001}
    ])

    print('[*] Running pretraining...')

    for i in range(epochs):
        print('[i] Epoch %s' % (i + 1))
        for (xb, yb) in tqdm(dataloader):
            x = conv_layer(xb)
            scores = critic_layer(x)
            losses = loss_fn(scores, yb)
            optimizer.zero_grad()
            losses.mean().backward()
            optimizer.step()

    print('[*] Copying over model configuration...')
    model.policy_old.load_state_dict(model.policy.state_dict())

def train(log_dir, save_dir, train_for_epochs=-1):
    K_epochs = 100
    update_eps = 4
    log_every = 40
    pretrain_epochs = 10
    ppo_pretrain_epochs = 0
    
    K_epochs = 100
    clip = 0.2
    gamma = 0.99 # reward decay
    lr_actor = 0.0001
    lr_critic = 0.001
    lr_conv = 0.001

    width, height, n_mines = 50, 20, 199
    ppo = PPO(width, height, lr_actor, lr_critic, lr_conv, gamma, K_epochs, clip)
    gym = MinesweeperGym(width, height, n_mines)

    subdir = Path(input('Save subdir: '))

    log_dir = Path(log_dir) / subdir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    save_dir = Path(save_dir) / subdir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = lambda checkpoint_name: save_dir / ('PPO_%s.pth' % checkpoint_name)

    pretrain(ppo, width, height, n_mines, epochs=pretrain_epochs)

    print('[*] Starting Training')
    print('[H] K_epochs: %s\t update_eps: %s\t clip: %s\t gamma: %s' % (K_epochs, update_eps, clip, gamma))
    print('[H] Learning Rates - actor: %s\t critic: %s\t convolution: %s' % (lr_actor, lr_critic, lr_conv))
    print('[M] Minesweeper board: w %s, h%s, %s mines' % (width, height, n_mines))
    print('[P] Saving to %s, logging to %s' % (save_dir, log_dir))

    iterator = itertools.count() if train_for_epochs < 0 else range(train_for_epochs)

    now = time.time()
    eps_rewards, eps_actions, eps_terminated, eps_won = [], [], [], []
    for epoch in iterator:
        eps_reward, n_actions, terminated, won = rollout(ppo, gym, pretrain=epoch<ppo_pretrain_epochs)
        eps_rewards.append(eps_reward)
        eps_actions.append(n_actions)
        eps_terminated.append(terminated)
        eps_won.append(won)

        if epoch % update_eps == update_eps - 1:
            ppo.update()

        if epoch % log_every == log_every - 1:
            avg_reward = sum(eps_rewards) / log_every
            avg_actions = sum(eps_actions) / log_every
            rate_terminated = sum(1 if t else 0 for t in eps_terminated)
            rate_solved = sum(1 if eps_won[i] and eps_terminated[i] else 0 for i in range(log_every))
            rate_failed = sum(1 if not eps_won[i] and eps_terminated[i] else 0 for i in range(log_every))
            elapsed = time.time() - now

            print()
            print('[*] Epoch %s' % epoch)
            print('[*] Avg Reward: %s' % avg_reward)
            print('[*] Avg # Actions: %s' % avg_actions)
            print('[*] Termination Rate: %s' % rate_terminated)
            print('[*] Solve Rate: %s' % rate_solved)
            print('[*] Fail Rate: %s' % rate_failed)
            print('[*] Took %ss to run %s epochs' % (elapsed, log_every))

            save_to = checkpoint_path(epoch)
            ppo.save(save_to)
            print('[*] Saving to %s' % save_to)

            now = time.time()
            eps_rewards, eps_actions, eps_terminated = [], [], []

if __name__ == '__main__':
    train('training/training_logs', 'training/saved_models')