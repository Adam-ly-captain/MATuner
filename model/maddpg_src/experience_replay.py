from utils.log import Log

import numpy as np
import torch


class ExperienceBuffer:
    """
    经验回放缓冲区
    """

    def __init__(self, state_shape, action_shape, max_size: int = 1000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros((max_size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_shape), dtype=np.float32)
        Log().Debug(
            f"ExperienceBuffer: state_shape: {state_shape}, action_shape: {action_shape}, max_size: {max_size}"
        )

    def add(self, state, action, reward, next_state):
        self.states[self.ptr] = state.detach().numpy()
        self.actions[self.ptr] = action.detach().numpy()
        self.rewards[self.ptr] = reward  # 标量
        self.next_states[self.ptr] = next_state.detach().numpy()
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=64):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.tensor(self.states[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.float32),
            torch.tensor(self.rewards[idx], dtype=torch.float32),
            torch.tensor(self.next_states[idx], dtype=torch.float32),
        )

    def save(self, path="./model/maddpg_src/res/", filename="experience.npz"):
        np.savez(
            path + filename,
            states=self.states[: self.size],
            actions=self.actions[: self.size],
            rewards=self.rewards[: self.size],
            next_states=self.next_states[: self.size],
        )

    def load(self, path="./model/maddpg_src/res/", filename="experience.npz"):
        data = np.load(path + filename)
        loaded_states = data["states"]
        loaded_size = loaded_states.shape[0]
        self.states[:loaded_size] = loaded_states
        self.actions[:loaded_size] = data["actions"]
        self.rewards[:loaded_size] = data["rewards"]
        self.next_states[:loaded_size] = data["next_states"]
        self.size = loaded_size
        self.max_size = self.states.shape[0]
        self.ptr = self.size % self.max_size
        data.close()

    def get_all(self):
        return (
            self.states[: self.size],
            self.actions[: self.size],
            self.rewards[: self.size],
            self.next_states[: self.size],
        )

    def __len__(self):
        return self.size

    def __str__(self):
        return f"ExperienceBuffer: size: {self.size}, max_size: {self.max_size}"
