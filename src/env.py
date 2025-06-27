import gym
import gym.spaces
import torch
import numpy as np
from .minesweeper import Board

class MinesweeperGym:
    def __init__(self, w, h, n):
        super().__init__()
        self.shape = (w, h)
        self.board = Board(w, h)
        self.board.place_mines(n)


class MinesweeperNetwork(torch.nn.Module):
    def __init__(self, w, h):
        self.shape = (w, h)
        self.conv_layer = torch.nn.Conv2d(in_channels=9, out_channels=1, kernel_size=3, stride=1),
        self.mlp = [
            torch.nn.Linear((w - 1) * (h - 1), 500),
            torch.nn.Linear(500, 500),
            torch.nn.Linear(500, 50)
        ]
        self.x_net = torch.nn.Linear(50, w)
        self.y_net = torch.nn.Linear(50, h)
        self.action_net = torch.nn.Linear(50, 2)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.flatten(x)
        for mlp in self.mlp:
            x = torch.nn.ReLU(x)
            x = mlp(x)
        x_coord = torch.nn.Softmax(self.x_net(x))
        y_coord = torch.nn.Softmax(self.y_net(x))
        action = torch.nn.Softmax(self.action_net(x))
        return (x_coord, y_coord, action)