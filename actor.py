import torch
import numpy as np

from time import time
import os
import gc

from model import QNet
from replay_memory import NStepMemory, ReplayMemory
from env import PongEnv


def actor_process(actor_id, n_actors, shared_dict, device='cuda:0'):
    actor = Actor(actor_id, n_actors, shared_dict, device)
    actor.run()


class Actor:
    def __init__(self, actor_id, n_actors, shared_dict, device='cpu'):
        # params
        self.gamma = 0.99
        self.epsilon = 0.4 ** (1 + actor_id * 7 / (n_actors - 1))
        self.bootstrap_steps = 3
        self.alpha = 0.6
        self.priority_epsilon = 1e-6
        self.device = device
        self.actor_id = actor_id

        # path
        self.memory_path = os.path.join(
            './', 'logs', 'memory')

        # memory
        self.memory_size = 50000
        self.batch_size = 32
        self.action_repeat = 4
        self.n_stacks = 4
        self.burn_in_length = 10
        self.learning_length = 10
        self.overlap_length = 10
        self.eta = 0.9
        self.sequence_length = self.burn_in_length + self.learning_length
        self.stack_count = self.n_stacks // self.action_repeat
        self.memory_save_interval = 5
        self.episode_start_index = 0
        self.n_steps_memory = NStepMemory(self.bootstrap_steps, self.gamma)
        self.replay_memory = ReplayMemory(self.memory_size, self.batch_size, self.bootstrap_steps)

        # net
        self.shared_dict = shared_dict
        self.net_load_interval = 5
        self.net = QNet(self.device).to(self.device)
        self.target_net = QNet(self.device).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())

        # env
        self.env = PongEnv(self.action_repeat, self.n_stacks)
        self.episode_reward = 0
        self.n_episodes = 0
        self.n_steps = 0
        self.memory_count = 0
        self.state = self.env.reset()
    
    def run(self):
        while True:
            self.step()

    def step(self):
        state = self.state
        action, q_value, h, c, target_q_value, target_h, target_c = self.select_action(state)
        q_value = q_value.detach().cpu().numpy()
        target_q_value = target_q_value.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)
        self.episode_reward += reward
        self.n_steps += 1

        self.n_steps_memory.add(q_value, state[-self.action_repeat:], h, c, target_h, target_c, action, reward, self.stack_count)
        if self.stack_count > 1:
            self.stack_count -= 1
        
        if self.n_steps > self.bootstrap_steps:
            pre_q_value, state, h, c, target_h, target_c, action, reward, stack_count = self.n_steps_memory.get()
            priority = self.calc_priority(pre_q_value, action, reward, q_value, target_q_value, done)
            self.replay_memory.add(state, h, c, target_h, target_c, action, reward, done, stack_count, priority)
            self.memory_count += 1
        self.state = next_state.copy()

        if done:
            while self.n_steps_memory.size > 0:
                pre_q_value, state, h, c, target_h, target_c, action, reward, stack_count = self.n_steps_memory.get()
                priority = self.calc_priority(pre_q_value, action, reward, q_value, target_q_value, done)
                self.replay_memory.add(state, h, c, target_h, target_c, action, reward, done, stack_count, priority)
                self.memory_count += 1
            self.reset()
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_value, h, c = self.net(state, True)
            target_q_value, target_h, target_c = self.target_net(state, True)
        if np.random.random() < self.epsilon:
            action = np.random.randint(6)
        else:
            action = q_value.argmax().item()

        return action, q_value, h, c, target_q_value, target_h, target_c
    
    def reset(self):
        if self.n_episodes % 1 == 0:
            print('episodes:', self.n_episodes, 'actor_id:', self.actor_id, 'return:', self.episode_reward)

        self.net.reset()
        self.target_net.reset()
        self.set_seq_start_index()
        self.state = self.env.reset()
        self.episode_start_index = self.replay_memory.index
        self.episode_reward = 0
        self.n_episodes += 1
        self.n_steps = 0
        self.memory_count = 0
        self.stack_count = self.n_stacks // self.action_repeat

        # reset n_step memory
        self.n_steps_memory = NStepMemory(self.bootstrap_steps, self.gamma)

        # save replay memory
        if self.n_episodes % self.memory_save_interval == 0:
            self.replay_memory.save(self.memory_path, self.actor_id)
            self.replay_memory = ReplayMemory(self.memory_size, self.batch_size, self.bootstrap_steps)
            self.episode_start_index = 0
            gc.collect()
        
        # load net
        if self.n_episodes % self.net_load_interval == 0:
            self.load_model()
    
    def load_model(self):
        try:
            self.net.load_state_dict(self.shared_dict['net_state'])
            self.target_net.load_state_dict(self.shared_dict['target_net_state'])
        except:
            print('load error')

    def calc_priority(self, q_value, action, reward, next_q_value, target_next_q_value, done):
        q_value = q_value.reshape(-1)[action]
        target_next_q_value = target_next_q_value.reshape(-1)

        if done:
            target_q_value = reward
        else:
            next_action = next_q_value.argmax(-1)
            target_next_q_value = target_next_q_value[next_action]
            target_q_value = reward + (self.gamma**self.bootstrap_steps) * target_next_q_value
        priority = np.abs(q_value - target_q_value) + self.priority_epsilon
        priority = priority ** self.alpha
    
        return priority
    
    def set_seq_start_index(self):
        last_index = self.replay_memory.index
        start_index  = self.episode_start_index

        seq_start_index = [i for i in range(start_index, last_index-self.sequence_length, self.overlap_length)]
        seq_start_index.append(last_index - self.sequence_length)
        seq_start_index = np.array(seq_start_index)
        self.replay_memory.update_sequence_priority(seq_start_index)
        self.replay_memory.memory['is_seq_start'][seq_start_index] = 1
