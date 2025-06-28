from .minesweeper import Board
import random
import numpy as np
import torch
from tqdm import tqdm

def generate_pretraining_data(w, h, n, n_eps=10000, batch_size=64):
    board_states = []
    scores = []
    for _ in tqdm(range(n_eps)):
        board = Board(w, h)
        board.place_mines(n)
        board.safe_click()
        board_states.append(board.model_state())
        scores.append(board.surrogate_eval())
        
        while True:
            x, y = random.randint(0, w-1), random.randint(0, h - 1)
            if random.random() > 0.2:
                _, terminate = board.click(x, y)
            else:
                board.flag(x, y)
                terminate = False
            if terminate:
                break
            board_states.append(board.model_state())
            scores.append(board.surrogate_eval())

    board_states, scores = np.array(board_states), np.array(scores)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(board_states, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)