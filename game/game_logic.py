import random
import config
import numpy as np
from numba import njit


def new_game(n):
    matrix = np.zeros((n, n), dtype=np.int)
    matrix = add_two_or_four(matrix)
    matrix = add_two_or_four(matrix)
    return matrix


@njit
def add_two_or_four(mat):
    empty_cell_list = np.argwhere(mat == 0)
    index = random.randint(0, len(empty_cell_list) - 1)
    if random.randint(0, 9) < 9:
        mat[empty_cell_list[index][0]][empty_cell_list[index][1]] = 2
    else:
        mat[empty_cell_list[index][0]][empty_cell_list[index][1]] = 4
    return mat


@njit
def game_state(mat):
    # check for win cell
    if np.any(mat == 2048):
        return 'win'
    elif np.any(mat == 0):
        return 'not over'
    else:
        # check for same cells that touch each other
        for i in range(len(mat) - 1):
            # intentionally reduced to check the row on the right and below
            # more elegant to use exceptions but most likely this will be their solution
            for j in range(len(mat[0]) - 1):
                if mat[i][j] == mat[i + 1][j] or mat[i][j + 1] == mat[i][j]:
                    return 'not over'
        for k in range(len(mat) - 1):  # to check the left/right entries on the last row
            if mat[len(mat) - 1][k] == mat[len(mat) - 1][k + 1]:
                return 'not over'
        for j in range(len(mat) - 1):  # check up/down entries on last column
            if mat[j][len(mat) - 1] == mat[j + 1][len(mat) - 1]:
                return 'not over'

    return 'lose'


@njit
def transpose(mat):
    new_mat = mat.copy()
    return new_mat.transpose()


@njit
def reverse(mat):
    new_mat = mat.copy()
    return new_mat[:, ::-1]


@njit
def shuffle_to_left(mat):
    score = 0
    new_mat = mat.copy()
    done = False
    for i in range(mat.shape[0]):
        # There must be no cells or only 0s between pos_l, pos_r
        pos_l, pos_r = 0, 1
        while not np.all(new_mat[i] == 0) and pos_r < mat.shape[1]:
            # m,n>0; m!=n
            # 0, 0
            if new_mat[i][pos_l] == new_mat[i][pos_r] and new_mat[i][pos_r] == 0:
                pos_r += 1
            # 0, m
            elif new_mat[i][pos_l] == 0 and new_mat[i][pos_r] > 0:
                new_mat[i][pos_l] = new_mat[i][pos_r]
                new_mat[i][pos_r] = 0
                pos_r += 1
                done = True
            # m, m
            elif new_mat[i][pos_l] == new_mat[i][pos_r] and new_mat[i][pos_r] > 0:
                new_mat[i][pos_l] *= 2
                score += new_mat[i][pos_l]
                new_mat[i][pos_r] = 0
                pos_l += 1
                pos_r += 1
                done = True
            # m, 0
            elif new_mat[i][pos_l] > 0 and new_mat[i][pos_r] == 0:
                pos_r += 1
            # m, n
            elif 0 < new_mat[i][pos_l] != new_mat[i][pos_r] > 0:
                pos_l += 1
                pos_r = pos_l + 1
            else:
                print(new_mat[i])

    return new_mat, done, score


@njit
def up(game):
    # print("up")
    # return matrix after shifting up
    game = transpose(game)
    game, done, score = shuffle_to_left(game)
    game = transpose(game)
    return game, done, score


@njit
def down(game):
    # print("down")
    # return matrix after shifting down
    game = reverse(transpose(game))
    game, done, score = shuffle_to_left(game)
    game = transpose(reverse(game))
    return game, done, score


@njit
def left(game):
    # print("left")
    # return matrix after shifting left
    game, done, score = shuffle_to_left(game)
    return game, done, score


@njit
def right(game):
    # print("right")
    # return matrix after shifting right
    game = reverse(game)
    game, done, score = shuffle_to_left(game)
    game = reverse(game)
    return game, done, score


def get_possible_actions(game):
    actions_l = []
    commands = {
        "Up": up,
        "Down": down,
        "Left": left,
        "Right": right
    }
    for each_possible_move in ["Up", "Down", "Left", "Right"]:
        _, done, _ = commands[each_possible_move](game)
        if done:
            actions_l.append(1)
        else:
            actions_l.append(0)

    return actions_l


# Scoring:
# This is the scoring according to how many times the matrix breaks the monotone in all rows and columns.
# The larger the score is, the worse the matrix is.
@njit
def score_monotone(mat):
    return score_monotone_for_rows(mat) + score_monotone_for_rows(transpose(mat))


@njit
def score_monotone_for_rows(mat):
    rst_score = 0
    # For each row
    for each_row in mat:
        previous_item = 0
        previous_tone = 'unknown'
        for index, each_item in enumerate(each_row):
            if each_item != 0 and index > 0:
                if previous_item != 0 and previous_tone == 'unknown':
                    if previous_item > each_item:
                        previous_tone = 'dec'
                    elif previous_item < each_item:
                        previous_tone = 'inc'
                elif previous_tone == 'inc':
                    if previous_item > each_item:
                        rst_score += 1
                        previous_tone = 'dec'
                elif previous_tone == 'dec':
                    if previous_item < each_item:
                        rst_score += 1
                        previous_tone = 'inc'

            if each_item != 0:
                previous_item = each_item

    return rst_score


@njit
def score_number_of_empty_squares(mat):
    cnt_array = np.where(mat, 0, 1)
    return np.sum(cnt_array)


@njit
def get_general_score(mat):
    return score_monotone_for_rows(mat)
