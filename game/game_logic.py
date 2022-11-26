import random
import config
import numpy as np


def new_game(n):
    matrix = []
    for i in range(n):
        matrix.append([0] * n)
    matrix = add_two_or_four(matrix)
    matrix = add_two_or_four(matrix)

    return matrix


def add_two_or_four(mat):
    empty_cell_list = np.argwhere(np.array(mat) == 0)
    index = random.randint(0, len(empty_cell_list) - 1)
    if random.randint(0, 9) < 9:
        mat[empty_cell_list[index][0]][empty_cell_list[index][1]] = 2
    else:
        mat[empty_cell_list[index][0]][empty_cell_list[index][1]] = 4
    return mat


def game_state(mat):
    # check for win cell
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 2048:
                return 'win'
    # check for any zero entries
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                return 'not over'
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


def transpose(mat):
    new = []
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])
    return new


def reverse(mat):
    new = []
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0]) - j - 1])
    return new


def cover_up(mat):
    new = []
    for j in range(config.GRID_LEN):
        partial_new = []
        for i in range(config.GRID_LEN):
            partial_new.append(0)
        new.append(partial_new)
    done = False
    for i in range(config.GRID_LEN):
        count = 0
        for j in range(config.GRID_LEN):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return new, done


def merge(mat, done):
    step_score = 0
    for i in range(config.GRID_LEN):
        for j in range(config.GRID_LEN - 1):
            if mat[i][j] == mat[i][j + 1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j + 1] = 0
                step_score += mat[i][j] + mat[i][j + 1]
                done = True
    return mat, done, step_score


def up(game):
    # print("up")
    # return matrix after shifting up
    game = transpose(game)
    game, done = cover_up(game)
    game, done, step_score = merge(game, done)
    game = cover_up(game)[0]
    game = transpose(game)
    return game, done, step_score


def down(game):
    # print("down")
    # return matrix after shifting down
    game = reverse(transpose(game))
    game, done = cover_up(game)
    game, done, step_score = merge(game, done)
    game = cover_up(game)[0]
    game = transpose(reverse(game))
    return game, done, step_score


def left(game):
    # print("left")
    # return matrix after shifting left
    game, done = cover_up(game)
    game, done, step_score = merge(game, done)
    game = cover_up(game)[0]
    return game, done, step_score


def right(game):
    # print("right")
    # return matrix after shifting right
    game = reverse(game)
    game, done = cover_up(game)
    game, done, step_score = merge(game, done)
    game = cover_up(game)[0]
    game = reverse(game)
    return game, done, step_score


def get_possible_actions(game):
    actions_l = []
    commands = {
        config.KEY_UP: up,
        config.KEY_DOWN: down,
        config.KEY_LEFT: left,
        config.KEY_RIGHT: right
    }
    for each_possible_move in [config.KEY_UP, config.KEY_DOWN, config.KEY_LEFT, config.KEY_RIGHT]:
        next_matrix, done, _ = commands[each_possible_move](game)
        if done:
            actions_l.append(1)
        else:
            actions_l.append(0)

    return actions_l


# Scoring:
# This is the scoring according to how many times the matrix breaks the monotone in all rows and columns.
# The larger the score is, the worse the matrix is.
def score_monotone(mat):
    return score_monotone_for_rows(mat) + score_monotone_for_rows(transpose(mat))


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


def score_number_of_empty_squares(mat):
    cnt_array = np.where(mat, 0, 1)
    return np.sum(cnt_array)


def get_general_score(mat):
    return score_monotone_for_rows(mat)
