import numpy
import torch
from torch import optim
import numpy as np
import random

import config
from agent.dueling_net import CNNNet
from common.transaciton import Transition
from agent.memoryreplayer2 import Buffer


class Brain:
    def __init__(self, num_states, num_actions):
        # Prepare the device:
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f"Training on device {self.device}.")

        self.num_actions = num_actions

        # self.memory = ReplayMemory(config.CAPACITY)
        self.buffer = Buffer(num_states, 'priority', config.CAPACITY)

        n_in, n_mid_1, n_mid_2, n_out = int(
            np.sqrt(num_states)), config.MIDDLE_LAYER_1_SIZE, config.MIDDLE_LAYER_2_SIZE, num_actions
        self.main_q_network = CNNNet(n_in, n_out)
        self.target_q_network = CNNNet(n_in, n_out)

        # Mode to GPU
        self.main_q_network.to(device=self.device)
        self.target_q_network.to(device=self.device)

        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=config.LEARNING_RATE)

        # Tensors
        self.state_action_values = None
        self.expected_state_action_values = None
        self.state_batch = None
        self.action_batch = None
        self.batch = None
        self.next_states = None
        self.reward_batch = None
        self.tree_idx = None
        self.ISWeights = None

    # All of them should be Tensors
    def push(self, state, action, state_next, reward):
        state = state.to(device=self.device)
        action = action.to(device=self.device)
        state_next = state_next.to(device=self.device)
        reward = reward.to(device=self.device)

        # self.memory.push(state, action, state_next, reward)
        self.buffer.store(state, action, state_next, reward)

    def replay(self):

        # if len(self.memory) < config.BATCH_SIZE:
        # Don't learn if the buffer is not full
        if self.buffer.memory_counter < config.CAPACITY:
            return

        # self.batch, self.state_batch, self.action_batch, self.reward_batch, self.next_states, \
        # self.tree_idx, self.ISWeights = self.make_minibatch()

        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.next_states, \
        self.tree_idx, self.ISWeights = self.make_minibatch_all_directions()

        self.expected_state_action_values = self.get_expected_state_action_values()
        self.update_main_q_network()

    # The state must be the flattened log2 of the original matrix.
    # Action the direction code in tensor [[]]: 0 up, 1 down, 2 left, 3 right
    def decide_action(self, state, possible_actions, episode, is_random=False):
        if is_random:
            action = random.randrange(self.num_actions)
            return torch.LongTensor([[action]])
        else:
            state = state.to(device=self.device)
            epsilon = config.INITIAL_EPSILON * (1 - episode / config.NUM_EPISODES)
            if epsilon <= np.random.uniform(0, 1):
                self.main_q_network.eval()
                with torch.no_grad():
                    action = self.main_q_network(state)
                    action = action.view(-1).tolist()

                    if config.SKIP_IMPOSSIBLE_ACTION:
                        for i in range(self.num_actions):
                            if possible_actions[i] == 0:
                                action[i] = -1000

                    action = action.index(max(action))
            else:
                action = random.randrange(self.num_actions)
                if config.SKIP_IMPOSSIBLE_ACTION:
                    while possible_actions[action] == 0:
                        action = random.randrange(self.num_actions)

            action = torch.LongTensor([[action]])
            return action

    def make_minibatch(self):
        # transactions = self.memory.sample(config.BATCH_SIZE)
        transactions, (tree_idx, ISWeights) = self.buffer.sample(config.BATCH_SIZE)

        batch = Transition(*zip(*transactions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)

        return batch, state_batch, action_batch, reward_batch, next_states, tree_idx, ISWeights

    def make_minibatch_all_directions(self):
        transactions, (tree_idx, ISWeights) = self.buffer.sample(config.BATCH_SIZE)

        batch = Transition(*zip(*transactions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)

        new_tree_idx = tree_idx
        new_ISWeights = ISWeights
        for i in range(1, 8):
            new_tree_idx = numpy.append(new_tree_idx, tree_idx)
            new_ISWeights = numpy.append(new_ISWeights, tree_idx)

        state_batch = get_states_in_all_dir(state_batch)
        action_batch = get_actions_in_all_dir(action_batch, self.device)
        reward_batch = get_multiplied_rewards(reward_batch)
        next_states = get_states_in_all_dir(next_states)

        return batch, state_batch, action_batch, reward_batch, next_states, new_tree_idx, new_ISWeights

    def update_main_q_network(self):
        # Train mode:
        self.main_q_network.train()

        # loss = F.mse_loss(self.state_action_values, self.expected_state_action_values)
        # Need to convert to ndarrays here:
        abs_errors = (self.expected_state_action_values - self.state_action_values).abs()
        abs_errors = abs_errors.cpu().detach().numpy()

        self.buffer.update(self.tree_idx, abs_errors)
        loss = (torch.FloatTensor(self.ISWeights).to(device=self.device) * (self.expected_state_action_values
                                                                            - self.state_action_values).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def get_expected_state_action_values(self):
        # Evaluation mode:
        self.main_q_network.eval()
        self.target_q_network.eval()
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)
        next_state_values = self.target_q_network(self.next_states).detach()
        next_state_action_values = self.main_q_network(self.next_states)
        q_eval_argmax = next_state_action_values.max(1)[1].view(next_state_action_values.shape[0], 1)
        next_state_values = next_state_values.gather(1, q_eval_argmax).view(next_state_values.shape[0], 1)
        expected_state_action_values = self.reward_batch + config.GAMMA * next_state_values

        return expected_state_action_values


# Transform each single state into all 8 directions and concatenate.
# Input: a tensor of BATCH_SIZE*10*GRID_LEN*GRID_LEN
def get_states_in_all_dir(state):
    state_reshape_1 = torch.flip(state, [3])
    state_reshape_2 = torch.flip(state, [2])
    state_reshape_3 = torch.transpose(state, 2, 3)
    state_reshape_4 = torch.flip(state_reshape_2, [3])
    state_reshape_5 = torch.flip(state_reshape_3, [2])
    state_reshape_6 = torch.flip(state_reshape_3, [3])
    state_reshape_7 = torch.flip(state_reshape_5, [3])

    multiplied_state = torch.cat(
        [state, state_reshape_1, state_reshape_2, state_reshape_3, state_reshape_4, state_reshape_5, state_reshape_6,
         state_reshape_7], 0)
    return multiplied_state


# Input: a tensor of 128*1
def convert_each_action(dir_code, single_action):
    return config.ACTION_NUMBERS_IN_RESHAPE[dir_code][single_action]


def get_actions_in_all_dir(action, device):
    action_tensor_list = []
    action_ndarray = action.cpu()
    for i in range(0, 8):
        action_tensor = action_ndarray.apply_(lambda x: convert_each_action(i, x))
        action_tensor_list.append(action_tensor)

    multiplied_actions = torch.cat(action_tensor_list, 0)
    multiplied_actions = multiplied_actions.to(device=device)
    return multiplied_actions


def get_multiplied_rewards(reward_batch):
    reward_batch_list = []
    for i in range(0, 8):
        reward_batch_list.append(reward_batch)

    multiplied_reward = torch.cat(reward_batch_list, 0)
    return multiplied_reward
