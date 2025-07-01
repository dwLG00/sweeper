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
    
    def display(self, clicked=False, reveal=False):
        if self.uncovered or reveal:
            if self.is_mine():
                return '\033[31m' + MINE + '\033[0m'
            else:
                if clicked:
                    return '\033[32m' + str(self.value) + '\033[0m'
                return str(self.value)
        else:
            if self.is_flagged():
                return '\033[31m' + 'F' + '\033[0m'
            return COVERED

class Board:
    def __init__(self, w, h): # use [x][y] indexing
        self.width = w
        self.height = h
        self.board = [[Cell() for _ in range(h)] for _ in range(w)]
        self.uncovered = 0
        self.total_mines = 0

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

        self.total_mines = n

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
    
    def display(self, clicked=None):
        # annoying, we can't print row by row bc our coord system so we'll just iterate
        cx, cy = clicked if clicked else (-1, -1)
        for y in range(self.height):
            buffer = []
            for x in range(self.width):
                if x == cx and y == cy:
                    buffer.append(self.board[x][y].display(clicked=True))
                else:
                    buffer.append(self.board[x][y].display())
            print(''.join(buffer))

    def display_all(self):
        for y in range(self.height):
            buffer = []
            for x in range(self.width):
                buffer.append(self.board[x][y].display(reveal=True))
            print(''.join(buffer))

    def click(self, x, y):
        # Return # of cells uncovered, or -1 if uncovered mine
        cell = self.board[x][y]
        if not cell.is_uncovered():
            if cell.uncover():
                return -1, True
            acc = 1
            if cell.value == 0: # clear everything around it as well (guaranteed safe)
                for nx, ny in self.neighbor_coords(x, y):
                    if not self.board[nx][ny].is_uncovered():
                        r, terminated = self.click(nx, ny)
                        acc += r
                        if terminated:
                            return acc, terminated
            self.uncovered += acc
            return acc, self.uncovered + self.total_mines == self.width * self.height
        else:
            if cell.value > 0 and sum(1 if self.board[nx][ny].is_flagged() else 0 for (nx, ny) in self.neighbor_coords(x, y)) == cell.value:
                still_covered = list(self.board[nx][ny] for (nx, ny) in self.neighbor_coords(x, y) if not self.board[nx][ny].is_uncovered())
                if all(cell.uncover() for cell in still_covered):
                    amt = len(still_covered)
                    self.uncovered += amt
                    return amt, self.uncovered + self.total_mines == self.width * self.height
                else:
                    return -1, True
            # do nothing
            return 0, False
    
    def safe_clicks(self):
        candidates = []
        maximum = 8
        for x in range(self.width):
            for y in range(self.height):
                if not self.board[x][y].is_mine():
                    if self.board[x][y].value == maximum:
                        candidates.append((x, y))
                    elif self.board[x][y].value < maximum:
                        candidates = [(x, y)]
                        maximum = self.board[x][y].value
        return candidates

    def safe_click(self):
        # click somewhere guaranteed safe, with preference for lower-valued cells
        candidates = self.safe_clicks()
        x, y = random.choice(candidates)
        self.click(x, y)

    def flag(self, x, y):
        cell = self.board[x][y]
        if not cell.is_uncovered():
            cell.toggle_flag()
            return 1
        return 0
    
    def model_state(self):
        l = []
        l.append(list(
            list(cell.value / 8 if not cell.is_mine() else -1.0 for cell in col) for col in self.board
        ))
        l.append(list(
            list(1 if not cell.is_uncovered() else 0 for cell in col) for col in self.board
        ))
        l.append(list(
            list(1 if cell.is_flagged() else 0 for cell in col) for col in self.board
        ))
        return l
    
    def surrogate_eval(self):
        score = 0
        for x in range(self.width):
            for y in range(self.height):
                if self.board[x][y].is_uncovered() and not self.board[x][y].is_mine():
                    score += 1
                    if all(not self.board[nx][ny].is_uncovered() for (nx, ny) in self.neighbor_coords(x, y)):
                        score -= 20
                if self.board[x][y].is_flagged():
                    if all(not self.board[nx][ny].is_uncovered() for (nx, ny) in self.neighbor_coords(x, y)): # bad location
                        score -= 100
        return score / (self.width * self.height)
    
    def mask(self):
        for x in range(self.width):
            for y in range(self.height):
                if self.board[x][y].is_uncovered():
                    yield y * self.width + x