import torch
import numpy as np

import os

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
        self.net_path = os.path.join(
            './', 'logs', 'model', 'net.pt')
        self.target_net_path = os.path.join(
            './', 'logs', 'model', 'target_net.pt')

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
        self.memory_save_interval = 1
        self.n_steps_memory = NStepMemory(self.bootstrap_steps, self.gamma)
        self.replay_memory = ReplayMemory(self.memory_size, self.batch_size, self.bootstrap_steps)

        # net
        self.shared_dict = shared_dict
        self.net_load_interval = 5
        self.net = QNet(self.net_path).to(self.device)
        self.target_net = QNet(self.target_net_path).to(self.device)
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
        action = self.select_action(state)
        next_state, reward, done, _ = self.env.step(action)
        self.episode_reward += reward
        self.n_steps += 1

        self.n_steps_memory.add(state[-self.action_repeat:], action, reward, self.stack_count)
        if self.stack_count > 1:
            self.stack_count -= 1
        
        if self.n_steps > self.bootstrap_steps:
            state, action, reward, stack_count = self.n_steps_memory.get()
            self.replay_memory.add(state, action, reward, done, stack_count)
            self.memory_count += 1
        self.state = next_state.copy()

        if done:
            while self.n_steps_memory.size > 0:
                state, action, reward, stack_count = self.n_steps_memory.get()
                self.replay_memory.add(state, action, reward, done, stack_count)
                self.memory_count += 1
            self.reset()
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(6)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_val = self.net(state)
                action = q_val.argmax().item()
        return action
    
    def reset(self):
        if self.n_episodes % 1 == 0:
            print('episodes:', self.n_episodes, 'actor_id:', self.actor_id, 'return:', self.episode_reward)

        self.calc_priority()
        self.state = self.env.reset()
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
        
        # load net
        if self.n_episodes % self.net_load_interval == 0:
            self.load_model()
    
    def load_model(self):
        try:
            self.net.load_state_dict(self.shared_dict['net_state'])
            self.target_net.load_state_dict(self.shared_dict['target_net_state'])
        except:
            print('load error')
    
    def calc_priority(self):
        last_index = self.replay_memory.size
        start_index  = last_index - self.memory_count

        batch, index = self.replay_memory.indexing_sample(start_index, last_index, self.device)
        batch_size = batch['state'].shape[0]
        priority = np.zeros(batch_size, dtype=np.float32)

        mini_batch_size = 500
        for start_index in range(0, batch_size, mini_batch_size):
            last_index = min(start_index + mini_batch_size, batch_size)
            mini_batch = dict()
            for key in batch.keys():
                if key in ['reward', 'done']:
                    mini_batch[key] = batch[key][start_index: last_index]
                else:
                    mini_batch[key] = torch.tensor(batch[key][start_index: last_index]).to(self.device)
            mini_batch['action'] = mini_batch['action'].view(-1,1).long()

            with torch.no_grad():
                # q_value
                q_value = self.net(
                    mini_batch['state']).gather(1, mini_batch['action']).view(-1,1).cpu().numpy()

                # taget_q_value
                next_action = torch.argmax(self.net(
                    mini_batch['next_state']), 1).view(-1,1)
                next_q_value = self.target_net(
                    mini_batch['next_state']).gather(1, next_action).cpu().numpy()
            
            target_q_value = mini_batch['reward'] + (self.gamma**self.bootstrap_steps) * next_q_value * (1 - mini_batch['done'])
            delta = np.abs(q_value - target_q_value).reshape(-1) + self.priority_epsilon
            delta = delta ** self.alpha
            priority[start_index: last_index] = delta
        
        self.replay_memory.update_priority(index, priority)

        seq_start_index = [i for i in range(start_index, last_index-self.sequence_length, self.overlap_length)]
        seq_start_index.append(last_index - self.sequence_length)
        seq_start_index = np.array(seq_start_index)
        self.replay_memory.update_sequence_priority(seq_start_index)
        self.replay_memory.memory['is_seq_start'][seq_start_index] = 1