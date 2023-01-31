import torch
import config
import numpy as np
import math

def get_states_in_all_dir(state):
    state = state.view(config.BATCH_SIZE, config.GRID_LEN, config.GRID_LEN)
    state_reshape_1 = torch.flip(state, [2])
    state_reshape_2 = torch.flip(state, [1])
    state_reshape_3 = torch.transpose(state, 1, 2)
    state_reshape_4 = torch.flip(state_reshape_2, [2])
    state_reshape_5 = torch.flip(state_reshape_3, [1])
    state_reshape_6 = torch.flip(state_reshape_3, [2])
    state_reshape_7 = torch.flip(state_reshape_5, [2])

    multiplied_state = torch.cat(
        [state, state_reshape_1, state_reshape_2, state_reshape_3, state_reshape_4, state_reshape_5, state_reshape_6,
         state_reshape_7], 0)
    multiplied_state = multiplied_state.view(-1, config.GRID_LEN * config.GRID_LEN)
    return multiplied_state

def convert_each_action(dir, single_action):
    return config.ACTION_NUMBERS_IN_RESHAPE[dir][single_action]

def get_actions_in_all_dir(action):
    action_tensor_list = []
    action_ndarray = action.numpy()
    for i in range(0, 8):
        action_reshape = np.array(list(map(lambda x: convert_each_action(i, x[0]), action_ndarray)))
        action_tensor = torch.tensor(action_reshape).view(config.BATCH_SIZE, -1)
        action_tensor_list.append(action_tensor)

    multiplied_actions = torch.cat(action_tensor_list, 0)

    return multiplied_actions


# Input "cells" is a ndarray of GRID_LEN * GRID_LEN.
def cast_matrix_to_10channels(cells):
    expended_array = None
    for i in range(1, 11):
        temp_array = np.where(cells == math.pow(2, i), 1, 0)
        temp_array = np.expand_dims(temp_array, axis=0)
        if expended_array is None:
            expended_array = temp_array
        else:
            expended_array = np.append(expended_array, temp_array, axis=0)

    return expended_array




'''
in_state = torch.randint(0, 10, (128, 16))
get_states_in_all_dir(in_state)

in_action = torch.randint(0, 3, (128, 1))
get_actions_in_all_dir(in_action)
'''

test_array1 = np.array([i for i in range(1, 17)]).reshape(4, 4)
cast_matrix_to_10channels(test_array1)
