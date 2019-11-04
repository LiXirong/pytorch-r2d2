import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from collections import deque
import os
from time import time, sleep
import gc
import fasteners
import pickle


class NStepMemory(dict):
    def __init__(self, memory_size=3, gamma=0.99):
        self.memory_size = memory_size
        self.gamma = gamma

        self.q_value = deque(maxlen=memory_size)
        self.state = deque(maxlen=memory_size)
        self.hs = deque(maxlen=memory_size)
        self.cs = deque(maxlen=memory_size)
        self.target_hs = deque(maxlen=memory_size)
        self.target_cs = deque(maxlen=memory_size)
        self.action = deque(maxlen=memory_size)
        self.reward = deque(maxlen=memory_size)
        self.stack_count = deque(maxlen=memory_size)
    
    @property
    def size(self):
        return len(self.state)

    def add(self, q_value, state, hs, cs, target_hs, target_cs, action, reward, stack_count):
        self.q_value.append(q_value)
        self.state.append(state)
        self.hs.append(hs)
        self.cs.append(cs)
        self.target_hs.append(target_hs)
        self.target_cs.append(target_cs)
        self.action.append(action)
        self.reward.append(reward)
        self.stack_count.append(stack_count)

    def get(self):
        q_value = self.q_value.popleft()
        state = self.state.popleft()
        hs = self.hs.popleft()
        cs = self.cs.popleft()
        target_hs = self.target_hs.popleft()
        target_cs = self.target_cs.popleft()
        action = self.action.popleft()
        stack_count = self.stack_count.popleft()
        reward = sum([self.gamma ** i * r for i,r in enumerate(self.reward)])
        return q_value, state, hs, cs, target_hs, target_cs, action, reward, stack_count
    
    def is_full(self):
        return len(self.state) == self.memory_size


class ReplayMemory:
    def __init__(self, memory_size=100000, batch_size=32, n_step=3, state_size=(84,84), cell_size=256, action_repeat=4, n_stacks=4, alpha=0.4):
        self.index = 0
        self.memory_size = memory_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.state_size = (action_repeat,) + state_size
        self.action_repeat = action_repeat
        self.n_stacks = n_stacks // action_repeat
        self.alpha = alpha
        self.beta = 0.4
        self.beta_step = 0.00025 / 4
        self.eta = 0.9
        self.burn_in_length = 10
        self.learning_length = 10
        self.sequence_length = self.burn_in_length + self.learning_length

        self.memory = dict()
        self.memory['state'] = np.zeros((self.memory_size, *self.state_size), dtype=np.uint8)
        self.memory['hs_cs'] = np.zeros((self.memory_size, cell_size*2), dtype=np.float32)
        self.memory['target_hs_cs'] = np.zeros((self.memory_size, cell_size*2), dtype=np.float32)
        self.memory['action'] = np.zeros((self.memory_size, 1), dtype=np.int8)
        self.memory['reward'] = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.memory['done'] = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.memory['stack_count'] = np.zeros((self.memory_size,), dtype=np.int8)
        self.memory['priority'] = np.zeros((self.memory_size,), dtype=np.float32)
        self.memory['sequence_priority'] = np.zeros((self.memory_size,), dtype=np.float32)
        self.memory['is_seq_start'] = np.zeros((self.memory_size,), dtype=np.uint8)
        self.arange = np.arange(self.memory_size)

    @property
    def size(self):
        return min(self.index, self.memory_size)
    
    def add(self, state, hs, cs, target_hs, target_cs, action, reward, done, stack_count, priority):
        index = self.index % self.memory_size
        self.memory['state'][index] = state * 255
        self.memory['hs_cs'][index, :self.cell_size] = hs
        self.memory['hs_cs'][index, self.cell_size:] = cs
        self.memory['target_hs_cs'][index, :self.cell_size] = target_hs
        self.memory['target_hs_cs'][index, self.cell_size:] = target_cs
        self.memory['action'][index] = action
        self.memory['reward'][index] = reward
        self.memory['done'][index] = 1 if done else 0
        self.memory['stack_count'][index] = stack_count
        self.memory['priority'][index] = priority
        self.index += 1
    
    def extend(self, memory):
        start_index = self.index % self.memory_size
        last_index = (start_index + memory['state'].shape[0]) % self.memory_size
        if start_index < last_index:
            index = [i for i in range(start_index, last_index)]
        else:
            index = [i for i in range(start_index, self.memory_size)] + [i for i in range(last_index)]
        index = np.array(index)
        
        for key in self.memory.keys():
            self.memory[key][index] = memory[key]

        self.index += memory['state'].shape[0]
    
    def fit(self):
        for key in self.memory.keys():
            self.memory[key] = self.memory[key][:self.size]
    
    def save(self, path, actor_id):
        path = os.path.join(path, f'memory{actor_id}.pt')
        lock = fasteners.InterProcessLock(path)

        while True:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                if lock.acquire(blocking=False):
                    try:
                        memory = torch.load(path, map_location=lambda storage, loc: storage)
                        self.extend(memory)
                        self.fit()
                        torch.save(self.memory, path)
                    except:
                        os.remove(path)
                    lock.release()
                    gc.collect()
                    return
            else:
                if lock.acquire(blocking=False):
                    try:
                        self.fit()
                        torch.save(self.memory, path)
                    except:
                        os.remove(path)
                    lock.release()
                    gc.collect()
                    return
            sleep(np.random.random()+2)

    
    def load(self, path, actor_id):
        path = os.path.join(path, f'memory{actor_id}.pt')
        lock = fasteners.InterProcessLock(path)

        while True:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                if lock.acquire(blocking=False):
                    try:
                        memory = torch.load(path, map_location=lambda storage, loc: storage)
                        self.extend(memory)
                    except:
                        pass
                    os.remove(path)
                    lock.release()
                    gc.collect()
                    return
                else:
                    sleep(np.random.random())
            return
    
    def update_priority(self, index, priority):
        self.memory['priority'][index] = priority
    
    def set_hs_cs(self, index, hs, cs, target_hs, target_cs):
        self.memory['hs_cs'][index, :self.cell_size] = hs
        self.memory['hs_cs'][index, self.cell_size:] = cs
        self.memory['target_hs_cs'][index, :self.cell_size] = target_hs
        self.memory['target_hs_cs'][index, self.cell_size:] = target_cs
    
    def update_sequence_priority(self, index, update_pre_next_seq_priority=False):
        for idx in index:
            indices = np.arange(idx, idx+self.sequence_length) % self.memory_size
            priority = self.memory['priority'][idx: idx+self.sequence_length]
            self.memory['sequence_priority'][idx] = self.eta * priority.max() + (1 - self.eta) * priority.mean()

            if update_pre_next_seq_priority:
                update = False
                for i in range(1, self.sequence_length+1):
                    if i < 0:
                        i = self.memory_size + i
                    if self.memory['is_seq_start'][(idx-i)%self.memory_size] == 1:
                        pre_idx = idx - i
                        update = True
                        break
                if update:
                    indices = np.arange(pre_idx, pre_idx+self.sequence_length) % self.memory_size
                    priority = self.memory['priority'][indices]
                    self.memory['sequence_priority'][pre_idx] = self.eta * priority.max() + (1 - self.eta) * priority.mean()
        
                update = False
                for i in range(1, self.sequence_length+1):
                    if self.memory['is_seq_start'][(idx+i)%self.memory_size] == 1:
                        next_idx = idx - i
                        update = True
                        break
                if update:
                    indices = np.arange(next_idx, next_idx+self.sequence_length) % self.memory_size
                    priority = self.memory['priority'][indices]
                    self.memory['sequence_priority'][next_idx] = self.eta * priority.max() + (1 - self.eta) * priority.mean()

    def get_stacked_state(self, index):
        stack_count = self.memory['stack_count'][index]
        start_index = index - (self.n_stacks - stack_count)
        if start_index < 0:
            start_index = self.memory_size + start_index
        stack_index = [start_index for _ in range(stack_count)] + [(start_index+1+i)%self.memory_size for i in range(self.n_stacks-stack_count)]
        stacked_state = np.concatenate([self.memory['state'][i] for i in stack_index])
        return stacked_state

    def sample(self, device='cpu'):
        seq_start_index = self.arange[self.memory['is_seq_start']==1]
        priority = self.memory['sequence_priority'][seq_start_index]
        seq_index = WeightedRandomSampler(
            priority / np.sum(priority),
            self.batch_size,
            replacement=True)
        seq_index = np.array(list(seq_index))
        seq_index = seq_start_index[seq_index]
        next_seq_index = (seq_index + self.n_step) % self.memory_size

        batch = dict()
        batch['state'] = [np.stack([self.get_stacked_state(i%self.memory_size) for i in seq_index+s]) for s in range(self.sequence_length)]
        batch['next_state'] = [np.stack([self.get_stacked_state(i%self.memory_size) for i in next_seq_index+s]) for s in range(self.sequence_length)]
        batch['hs'] = self.memory['hs_cs'][seq_index, :self.cell_size]
        batch['cs'] = self.memory['hs_cs'][seq_index, self.cell_size:]
        batch['target_hs'] = self.memory['target_hs_cs'][next_seq_index, :self.cell_size]
        batch['target_cs'] = self.memory['target_hs_cs'][next_seq_index, self.cell_size:]
        batch['action'] = [self.memory['action'][(seq_index+self.burn_in_length+s)%self.memory_size] for s in range(self.learning_length)]
        batch['reward'] = [self.memory['reward'][(seq_index+self.burn_in_length+s)%self.memory_size] for s in range(self.learning_length)]
        batch['done'] = [self.memory['done'][(seq_index+self.burn_in_length+s)%self.memory_size] for s in range(self.learning_length)]
        for key in batch.keys():
            if key not in ['hs', 'cs', 'target_hs', 'target_cs']:
                batch[key] = np.stack(batch[key])
        index = np.stack([(seq_index+s)%self.memory_size for s in range(self.sequence_length)])


        for key in batch.keys():
            batch[key] = np.stack(batch[key])
        index = np.stack([(seq_index+s)%self.memory_size for s in range(self.sequence_length)])

        for key in ['state', 'next_state']:
            batch[key] = batch[key].astype(np.float32) / 255.
        
        for key in batch.keys():
            batch[key] = torch.FloatTensor(batch[key]).to(device)
        batch['action'] = batch['action'].long()

        return batch, seq_index, index
    
    def indexing_sample(self, start_index, last_index, device='cpu'):
        index = np.arange(start_index, last_index) % self.memory_size
        next_index = (index + self.n_step) % self.memory_size

        batch = dict()
        batch['state'] = np.stack([[self.get_stacked_state(i)] for i in index])
        batch['next_state'] = np.stack([[self.get_stacked_state(i%self.memory_size)] for i in next_index])
        batch['action'] = self.memory['action'][index]
        batch['reward'] = self.memory['reward'][index]
        batch['done'] = self.memory['done'][index]

        batch['state'] = batch['state'].astype(np.float32) / 255.
        batch['next_state'] = batch['next_state'].astype(np.float32) / 255.
        return batch, index
