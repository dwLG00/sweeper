from src.minesweeper import Board
from src.model import PPO
import torch
import time

def test(model: PPO, t=1.5, mask=False):
    w, h = 50, 20
    board = Board(w, h) # default board settings
    board.place_mines(199)
    board.safe_click()

    print(board.display())
    while True:
        state = board.model_state()
        raw_val = model.select_action(state, mask=board.mask() if mask else None)
        x, y = raw_val % w, raw_val // w
        _, terminated = board.click(x, y)
        print(board.display(clicked=(x, y)))
        time.sleep(t)
        if terminated:
            break

if __name__ == '__main__':
    model_path = 'training/saved_models/fourteenth_serious_run/PPO_5239.pth'
    model = PPO(50, 20, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01) #only the first 2 args matter
    model.load(model_path)
    while True:
        test(model, t=0.5, mask=True)
