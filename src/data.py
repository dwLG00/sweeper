from .minesweeper import Board
import random
import numpy as np
import torch
from tqdm import tqdm

def generate_pretraining_data(w, h, n, n_eps=10000, n_midgame=10000, click_ratio=0.8, batch_size=64):
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
            _, terminate = board.click(x, y)
            if terminate:
                break
            board_states.append(board.model_state())
            scores.append(board.surrogate_eval())
    
    for _ in tqdm(range(n_midgame)):
        for (state, score) in generate_midgame(w, h, n, click_ratio=click_ratio, take_every=3, trajectory=True):
            board_states.append(state)
            scores.append(score)

    board_states, scores = np.array(board_states), np.array(scores)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(board_states, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def generate_midgame(w, h, n, click_ratio=0.8, take_every=1, trajectory=False):
    '''Generate boards where all zeros are clicked'''
    board = Board(w, h)
    board.place_mines(n)
    safe_click_coords = board.safe_clicks()
    n_keep = int(len(safe_click_coords) * click_ratio)
    random.shuffle(safe_click_coords)
    safe_click_coords = safe_click_coords[:n_keep]
    for i, (x, y) in enumerate(safe_click_coords):
        if not board.board[x][y].is_uncovered():
            board.click(x, y)
            if trajectory and i % take_every == 0:
                yield (board.model_state(), board.surrogate_eval())
    yield (board.model_state(), board.surrogate_eval())