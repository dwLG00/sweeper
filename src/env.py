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
        x, y, a = torch.split(action, [w, h, 2])
        x = torch.argmax(x)
        y = torch.argmax(y)
        a = torch.argmax(a)

        if a == 0: # click
            r, terminate = self.board.click(x, y)
            state = self.get_state()
            if r == -1: # clicked a mine
                return -w*h, state, True
            else:
                if r == 0: # did nothing
                    return -1, state, False
                if terminate:
                    r += w * h
                return r, state, terminate
            
        if a == 1:
            r = self.board.flag(x, y)
            state = self.get_state()
            return r - 1, state, False
        
    def reset(self):
        self.board = Board(self.width, self.height)
        self.board.place_mines(self.n_mines)
        self.board.safe_click() # click somewhere guaranteed safe
        return self.get_state()