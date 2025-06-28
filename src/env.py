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
        raw_val = action[0]
        x, y, a = raw_val % w, (raw_val // w) % h, raw_val // (w * h)

        if a == 0: # click
            prev_hidden = self.board.uncovered
            r, terminate = self.board.click(x, y)
            state = self.get_state()
            if r == -1: # clicked a mine
                return state, -2*w*h, True
            else:
                if r == 0: # did nothing
                    return state, -1, False
                if terminate:
                    r += w * h
                ratio = 1 + r / prev_hidden
                return state, r * ratio, terminate
            
        if a == 1:
            r = self.board.flag(x, y)
            state = self.get_state()
            score = -100 if r == 0 else -5
            return state, score, False
        
    def early_epoch_step(self, action: torch.Tensor):
        # run for first n steps as pretraining
        w, h = self.shape
        raw_val = action[0]
        x, y, a = raw_val % w, (raw_val // w) % h, raw_val // (w * h)

        if a == 0:
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
        if a == 1: # discourage flagging
            r = self.board.flag(x, y)
            state = self.get_state()
            return state, -w*h / 10, False
        
    def reset(self):
        w, h = self.shape
        self.board = Board(w, h)
        self.board.place_mines(self.n_mines)
        self.board.safe_click() # click somewhere guaranteed safe
        return self.get_state()