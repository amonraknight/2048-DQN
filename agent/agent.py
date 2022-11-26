from agent.brain import Brain
from common.module_saveload import ModuleSaveLoad


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)
        self.saveLoader = ModuleSaveLoad()

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, possible_actions, episode, is_random=False):
        action = self.brain.decide_action(state, possible_actions, episode, is_random)
        return action

    # All parameters should be in tensor.
    def memorize(self, state, action, state_next, reward):
        self.brain.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()

    def save_network(self, episode=0):
        self.saveLoader.save_module(self.brain.main_q_network, episode)

    def read_latest_module(self):
        module_backup, previous_episode = self.saveLoader.load_module()
        if module_backup and previous_episode:
            self.brain.main_q_network.load_state_dict(module_backup)
            self.brain.update_target_q_network()

        return previous_episode


