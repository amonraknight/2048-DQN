import numpy as np
import math
import copy
import torch
from numba import njit
from stopwatch import Stopwatch

import game_logic as logic
import config as c
from agent.agent import Agent


class Game2048NoDisplay:
    def __init__(self):
        self.matrix = logic.new_game(c.GRID_LEN)
        self.step_count = 0
        self.total_score = 0

        # DQN items
        num_states = c.GRID_LEN * c.GRID_LEN
        num_actions = len(c.ACTION_NUMBERS)
        self.agent = Agent(num_states, num_actions)

        self.commands = {
            0: logic.up,
            1: logic.down,
            2: logic.left,
            3: logic.right
        }

        self.stop_watch = Stopwatch()

    def observe(self):
        # number of each cell
        cells = np.array(self.matrix).flatten()
        cells = np.array(list(map(lambda x: math.log2(max(x, 1)) / 16, cells)))
        # game is over or not. 'win', 'lose', 'not over'
        status = logic.game_state(self.matrix)
        # possible actions
        matrix_copy = copy.deepcopy(self.matrix)
        action_l = logic.get_possible_actions(matrix_copy)
        return cells, status, action_l

    def reset(self):
        self.step_count = 0
        self.total_score = 0
        self.matrix = logic.new_game(c.GRID_LEN)
        return self.observe()

    def dqn_train(self):
        print('DQN solution training start...')
        complete_episodes = 0
        starting_episode = 0

        # Find if there are half-way modules.
        previous_episode = self.agent.read_latest_module()
        if previous_episode:
            starting_episode = previous_episode + 1

        # Each episode:
        for episode in range(starting_episode, c.NUM_EPISODES + 1):
            # Reset the game:
            cells, status, action_l = self.reset()

            # Prepare the state in tensor[[]]
            state = torch.from_numpy(cells).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            step = 0
            invalid_step_count = c.INVALID_STEP_TOLERATE
            while status == 'not over':
                action = self.agent.get_action(state, action_l, episode, is_random=invalid_step_count <= 0)

                if action_l[action] == 1:
                    # Commit the step: tensor.item converts the tensor with a single item to a single value.
                    state_next, status, action_l, reward = self.step(action.item())
                    step += 1
                    invalid_step_count = c.INVALID_STEP_TOLERATE
                else:
                    invalid_step_count -= 1
                    state_next = cells
                    reward = torch.FloatTensor([[c.INVALID_STEP_SCORE]])

                state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                state_next = torch.unsqueeze(state_next, 0)

                self.agent.memorize(state, action, state_next, reward)
                # Update the neronet after a configured interval.
                if episode % c.TRAIN_INTERVAL == 0:
                    self.agent.update_q_function()

                state = state_next

                if status == 'win' or status == 'lose':
                    if status == 'win':
                        complete_episodes += 1

                    elif status == 'lose':
                        complete_episodes = 0

                    print('%d Episode: Finished after %d steps in status %s.' % (episode, step, status))
                    if episode % c.TARGET_NET_UPDATE_INTERVAL == 0:
                        self.agent.update_target_q_function()
                    break

            if (episode > 0 and episode % c.BACKUP_INTERVAL == 0) or complete_episodes >= c.ACCEPT_THRESHOLD:
                self.agent.save_network(episode)
                if complete_episodes >= c.ACCEPT_THRESHOLD:
                    print('{} successful episodes.'.format(c.ACCEPT_THRESHOLD))
                    break

    # actions: 0 up, 1 down, 2 left, 3 right
    def step(self, action):
        reward = 0.0
        if 'mono-sequential' in c.REWARD_STRATEGY:
            reward = reward + logic.get_general_score(self.matrix)

        self.matrix, done, step_score = self.commands[action](self.matrix)
        self.total_score += step_score

        if 'mono-sequential' in c.REWARD_STRATEGY:
            reward = (reward - logic.get_general_score(self.matrix)) / 5

        if 'merged cells' in c.REWARD_STRATEGY:
            reward = reward + math.log2(max(step_score, 1))
        reward = torch.FloatTensor([[max(reward, 0)]])

        # Add a random cell.
        self.matrix = logic.add_two_or_four(self.matrix)

        observation, status, action_l = self.observe()
        return observation, status, action_l, reward

    def dqn_solve(self):
        c.SKIP_IMPOSSIBLE_ACTION = True
        total_steps = 0
        total_score = 0
        win_count = 0

        # Find if there are half-way modules.
        self.agent.read_latest_module()
        print('DQN solution training start...')
        self.stop_watch.start()

        # Each episode:
        for episode in range(c.TEST_ROUND):
            # Reset the game:
            cells, status, action_l = self.reset()

            # Prepare the state in tensor[[]]
            state = torch.from_numpy(cells).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            step = 0
            invalid_step = False
            while status == 'not over':
                action = self.agent.get_action(state, action_l, c.NUM_EPISODES, is_random=invalid_step)

                if action_l[action] == 1:
                    # Commit the step: tensor.item converts the tensor with a single item to a single value.
                    state_next, status, action_l, reward = self.step(action.item())
                    step += 1
                    invalid_step = False
                else:
                    invalid_step = True
                    state_next = cells

                state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                state_next = torch.unsqueeze(state_next, 0)

                state = state_next

                if status == 'win' or status == 'lose':
                    print('%d Episode: Finished after %d steps in status %s score %d.'
                          % (episode, step, status, self.total_score))
                    total_steps += step

                    total_score += self.total_score

                    if status == 'win':
                        win_count += 1
                    break

        self.stop_watch.stop()
        time_cost = round(self.stop_watch.duration, 0)

        print('Average steps {}, average score {}, {} in win. Time cost {}'.format(str(total_steps / c.TEST_ROUND),
                                                                                   str(total_score / c.TEST_ROUND),
                                                                                   str(win_count), str(time_cost)))


if __name__ == "__main__":
    game = Game2048NoDisplay()
    # game.dqn_train()
    game.dqn_solve()
