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
    
    def mask(self):
        return self.board.mask()

    def step(self, action: torch.Tensor):
        w, h = self.shape
        raw_val = action[0]
        x, y = raw_val % w, raw_val // w

        prev_hidden = self.board.uncovered
        r, terminate = self.board.click(x, y)
        state = self.get_state()
        if r == -1: # clicked a mine
            return state, 0, True, False
        else:
            if r == 0: # did nothing
                return state, -5, False, False
            if terminate:
                return state, 100, True, True
            #ratio = 1 + r / prev_hidden
            has_uncovered_neighbors = any(self.board.board[nx][ny].is_uncovered() for (nx, ny) in self.board.neighbor_coords(x, y))
            return state, 1 if has_uncovered_neighbors else -5, False, False
        
    def early_epoch_step(self, action: torch.Tensor):
        # run for first n steps as pretraining
        w, h = self.shape
        raw_val = action[0]
        x, y = raw_val % w, raw_val // w

    
        r, terminate = self.board.click(x, y)
        state = self.get_state()
        if r == -1: # don't penalize clicking mines, if the mine that was clicked was next to uncovered cell
            if any(self.board.board[nx][ny].is_uncovered() for (nx, ny) in self.board.neighbor_coords(x, y)):
                return state, 0, True
            else:
                return state, -w*h, True
        else:
            if r == 0: # heavily penalize doing nothing
                return state, -w*h / 10, False
            if terminate:
                r += w * h
            return state, r * 10, terminate # inflate clicking on a non-mine
        
    def reset(self):
        w, h = self.shape
        self.board = Board(w, h)
        self.board.place_mines(self.n_mines)
        self.board.safe_click() # click somewhere guaranteed safe
        return self.get_state()