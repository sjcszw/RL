import random

from collections import namedtuple

# a named tuple representing a single transition in our environment. 
# It essentially maps (state, action) pairs to their (next_state, reward)
# result, with the state being the screen difference 
Transition = namedtuple('Transition',
                        ('state_', 'action_', 'next_state_', 'reward_', 'done_'))


class ReplayMemory(object):
    ''' a cyclic buffer of bounded size that holds the 
    transitions observed recently. 
    '''
    def __init__(self, capacity):
        self.capacity = capacity  # largest # memory
        self.memory = []
        self.position = 0  # count length of memory

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # sample a batch from the buffer

    def __len__(self):
        return len(self.memory)
