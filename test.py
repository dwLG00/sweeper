from src.minesweeper import Board
from src.model import PPO
import torch
import time

def test(model: PPO, t=1.5):
    w, h = 50, 20
    board = Board(w, h) # default board settings
    board.place_mines(299)
    board.safe_click()

    print(board.display())
    while True:
        state = board.model_state()
        raw_val = model.select_action(state)
        x, y, a = raw_val % w, (raw_val // w) % h, raw_val // (w * h)
        if a == 0:
            _, terminated = board.click(x, y)
        elif a == 1:
            board.flag(x, y)
            terminated = False
        print(board.display(clicked=(x, y)))
        if terminated:
            break
        time.sleep(t)

if __name__ == '__main__':
    model_path = 'training/saved_models/tenth_serious_run/PPO_27519.pth'
    model = PPO(50, 20, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01) #only the first 2 args matter
    model.load(model_path)
    while True:
        test(model, t=0.5)
