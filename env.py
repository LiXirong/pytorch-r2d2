import cv2
import numpy as np
import gym

from collections import deque


def preprocess(frame):
    frame = np.uint8(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (84,84)))
    frame = np.reshape(frame, (1, 84,84))
    return frame


class PongEnv:
    def __init__(self, action_repeat=4, n_stacks=4):
        self.action_repeat = action_repeat
        self.n_stacks = n_stacks
        self.frame_queue = deque(maxlen=self.n_stacks)
        self.env = gym.make('Pong-v0')

    def step(self, action):
        state = []
        total_reward = 0
        for _ in range(self.action_repeat):
            frame, reward, done, info = self.env.step(action)
            self.frame_queue.append(preprocess(frame))
            total_reward += reward
            if done:
                break
        state = np.concatenate(list(self.frame_queue)).astype(np.float32) / 255.
        return state, total_reward, done, info

    def reset(self):
        _ = self.env.reset()
        for _ in range(np.random.randint(1,5)):
            state, _, _, _ = self.step(self.env.action_space.sample())
        return state
