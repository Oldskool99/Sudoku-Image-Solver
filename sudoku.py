class EntryData:
    """
    Entry Data for Board
    """

    def __init__(self, row, col, num):
        self.row = row
        self.col = col
        self.choices = num

    def set_data(self, row, col, num):
        """
        Set Board Data
        """
        self.row = row
        self.col = col
        self.choices = num


def solve_sudoku(matrix):
    """
    Main function which calls further functions
    """
    cont = [True]
    for i in range(9):
        for j in range(9):
            if not can_be_correct(matrix, i, j):
                return
    sudoku_helper(matrix, cont)


def sudoku_helper(matrix, cont):
    """
    Sudoku helper function
    """
    if not cont[0]:
        return

    best_candidate = EntryData(-1, -1, 100)
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                num_choices = count_choices(matrix, i, j)
                if best_candidate.choices > num_choices:
                    best_candidate.set_data(i, j, num_choices)

    if best_candidate.choices == 100:
        cont[0] = False
        return

    row = best_candidate.row
    col = best_candidate.col

    for j in range(1, 10):
        if not cont[0]:
            return

        matrix[row][col] = j

        if can_be_correct(matrix, row, col):
            sudoku_helper(matrix, cont)

    if not cont[0]:
        return
    matrix[row][col] = 0


def count_choices(matrix, i, j):
    """
    Count possible choices
    """
    can_pick = [True, True, True, True, True, True,
                True, True, True, True]

    for k in range(9):
        can_pick[matrix[i][k]] = False

    for k in range(9):
        can_pick[matrix[k][j]] = False

    row = i // 3
    col = j // 3
    for row_num in range(row*3, row*3+3):
        for col_num in range(col*3, col*3+3):
            can_pick[matrix[row_num][col_num]] = False

    count = 0
    for k in range(1, 10):
        if can_pick[k]:
            count += 1

    return count


def can_be_correct(matrix, row, col):
    """
    Check if the puzzle can be solved
    """
    for col_num in range(9):
        if matrix[row][col] != 0 and col != col_num and matrix[row][col] == matrix[row][col_num]:
            return False

    for row_num in range(9):
        if matrix[row][col] != 0 and row != row_num and matrix[row][col] == matrix[row_num][col]:
            return False

    row_num = row // 3
    col_num = col // 3
    for i in range(row_num*3, row_num*3+3):
        for j in range(col_num*3, col_num*3+3):
            if row != i and col != j and matrix[i][j] != 0 and matrix[i][j] == matrix[row][col]:
                return False

    return True


def all_board_non_zero(matrix):
    """
    Check if board contains any zero values
    """
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                return False
    return True
