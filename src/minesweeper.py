import random

MINE = '*'
COVERED = '█'

class Cell:
    def __init__(self):
        self.uncovered = False
        self.flagged = False
        self.value = 0

    def is_uncovered(self):
        return self.uncovered
    
    def uncover(self):
        self.uncovered = True
        return self.is_mine()

    def set_mine(self):
        self.value = -1

    def is_mine(self):
        return self.value == -1
    
    def toggle_flag(self):
        self.flagged = not self.flagged

    def is_flagged(self):
        return self.flagged
    
    def display(self):
        if self.uncovered:
            if self.is_mine():
                return MINE
            else:
                return str(self.value)
        else:
            return COVERED

class Board:
    def __init__(self, w, h): # use [x][y] indexing
        self.width = w
        self.height = h
        self.board = [[Cell() for _ in range(h)] for _ in range(w)]

    def place_mines(self, n):
        mine_locations = random.sample(list(range(self.width * self.height)), n)
        for idx in mine_locations:
            x = idx % self.width
            y = idx // self.width
            self.board[x][y].set_mine()
            # populate neighbors
            for nx, ny in self.neighbor_coords(x, y):
                if not self.board[nx][ny].is_mine():
                    self.board[nx][ny].value += 1

    def neighbor_coords(self, x, y):
        (xnleft, xnright, yntop, ynbottom) = (x != 0, x != self.width - 1, y != 0, y != self.height - 1)
        if xnleft:
            yield (x - 1, y)
            if yntop:
                yield (x - 1, y - 1)
            if ynbottom:
                yield (x - 1, y + 1)
        if xnright:
            yield (x + 1, y)
            if yntop:
                yield (x + 1, y - 1)
            if ynbottom:
                yield (x + 1, y + 1)
        if yntop:
            yield (x, y - 1)
        if ynbottom:
            yield (x, y + 1)
    
    def display(self):
        # annoying, we can't print row by row bc our coord system so we'll just iterate
        for y in range(self.height):
            buffer = []
            for x in range(self.width):
                buffer.append(self.board[x][y].display())
            print(''.join(buffer))

    def click(self, x, y):
        # Return # of cells uncovered, or -1 if uncovered mine
        cell = self.board[x][y]
        if not cell.is_uncovered():
            if cell.uncover():
                return -1
            acc = 1
            if cell.value == 0: # clear everything around it as well (guaranteed safe)
                for nx, ny in self.neighbor_coords(x, y):
                    if not self.board[nx][ny].is_uncovered():
                        acc += self.click(nx, ny)
            return acc
        else:
            if cell.value > 0 and sum(1 if self.board[nx][ny].is_flagged() else 0 for (nx, ny) in self.neighbor_coords(x, y)) == cell.value:
                if all(self.board[nx][ny].uncover() for (nx, ny) in self.neighbor_coords(x, y) if not self.board[nx][ny].is_flagged()):
                    return 9 - cell.value
                else:
                    return -1
            # do nothing
            return 0
        
    def flag(self, x, y):
        cell = self.board[x][y]
        if not cell.is_uncovered():
            cell.toggle_flag()
            return 1
        return 0