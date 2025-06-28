from .model import device
from .minesweeper import Board
import torch

class MinesweeperGym:
    def __init__(self, w, h, n):
        super().__init__()
        self.shape = (w, h)
        self.n_mines = n
        self.board = Board(w, h)
        self.board.place_mines(n)

    def get_state(self):
        return torch.tensor(self.board.model_state(), dtype=torch.float32).unsqueeze(0).to(device)
    
    def step(self, action: torch.Tensor):
        w, h = self.shape
        x, y, a = action[0], action[1], action[2]

        if a == 0: # click
            r, terminate = self.board.click(x, y)
            state = self.get_state()
            if r == -1: # clicked a mine
                return state, -2*w*h, True
            else:
                if r == 0: # did nothing
                    return state, -1, False
                if terminate:
                    r += w * h
                return state, r, terminate
            
        if a == 1:
            r = self.board.flag(x, y)
            state = self.get_state()
            return state, r - 1, False
        
    def reset(self):
        w, h = self.shape
        self.board = Board(w, h)
        self.board.place_mines(self.n_mines)
        self.board.safe_click() # click somewhere guaranteed safe
        return self.get_state()