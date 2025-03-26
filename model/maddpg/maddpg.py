"""
MATuner: Multi-Agent Tuner
基于MADDPG实现的适用于分布式MySQL数据库的自动参数调优器
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from torch.nn import init, Parameter
from torch.autograd import Variable

from utils.log import Log
from model.maddpg.experience_replay import ExperienceBuffer


# code from https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py
class NoisyLinear(nn.Linear):
    """
    噪声网络
    """

    def __init__(self, in_features, out_features, sigma_init=0.05, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=True
        )  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(
            self, "sigma_weight"
        ):  # Only init after all params added (otherwise super().__init__() fails)
            init.uniform_(
                self.weight,
                -math.sqrt(3 / self.in_features),
                math.sqrt(3 / self.in_features),
            )
            init.uniform_(
                self.bias,
                -math.sqrt(3 / self.in_features),
                math.sqrt(3 / self.in_features),
            )
            init.constant_(self.sigma_weight, self.sigma_init)
            init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(
            input,
            self.weight + self.sigma_weight * Variable(self.epsilon_weight),
            self.bias + self.sigma_bias * Variable(self.epsilon_bias),
        )

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class Actor(nn.Module):
    """
    Actor network
    input: 单个agent的state
    output: 单个agent的action
    """

    def __init__(self, state_dim, action_dim, noisy=True):
        super(Actor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.BatchNorm1d(128),  # 批标准化
            # nn.LayerNorm(128),  # 层标准化
            nn.Dropout(p=0.2),  # 防止过拟合
            nn.ReLU(0.2),  # state都是正值，所以用relu，而不用leakyrelu
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            # nn.LayerNorm(128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            # nn.LayerNorm(64),
        )
        if noisy:
            # behavior policy by adding noise
            self.out = NoisyLinear(64, action_dim)
        else:
            self.out = nn.Linear(64, action_dim)

        # 用于映射连续动作空间. [-1, 1]
        self.act = nn.Tanh()

        self.weights_init()

    def forward(self, state):
        output = self.layers(state)
        output = self.out(output)
        output = self.act(output)

        return output

    def sample_noise(self):
        self.out.sample_noise()

    def remove_noise(self):
        self.out.remove_noise()

    def weights_init(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


# class ActorSample(nn.Module):
#     """
#     Actor Sample network(sample noise的版本, 没有batchnorm)
#     """

#     def __init__(self, state_dim, action_dim):
#         super(ActorSample, self).__init__()

#         self.layers = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             # nn.BatchNorm1d(128),
#             nn.LayerNorm(128),
#             nn.Dropout(p=0.2),
#             nn.ReLU(0.2),
#             nn.Linear(128, 128),
#             # nn.BatchNorm1d(128),
#             nn.LayerNorm(128),
#             nn.Dropout(p=0.2),
#             nn.ReLU(0.2),
#             nn.Linear(128, 64),
#             # nn.BatchNorm1d(64),
#             nn.LayerNorm(64),
#             # nn.Linear(64, action_dim),
#             NoisyLinear(64, action_dim),
#             nn.Tanh(),
#         )

#     def forward(self, state):
#         return self.layers(state)

#     def sample_noise(self):
#         self.out.sample_noise()

#     def remove_noise(self):
#         self.out.remove_noise()


class Critic(nn.Module):
    """
    Critic network
    input: n个agent的state和action
    output: q value
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.BatchNorm1d(128),  # batch size 必须大于1
            # nn.LayerNorm(128),
            nn.Dropout(p=0.2),
            nn.ReLU(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            # nn.LayerNorm(128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128, 1),  # 输出q value
        )
        
        self.weights_init()

    def forward(self, state, action):
        return self.layers(torch.cat([state, action], 1))
      
    def weights_init(self):
      for m in self.layers:
          if isinstance(m, nn.Linear):
              nn.init.xavier_uniform_(m.weight)
              nn.init.constant_(m.bias, 0)


# class CriticLow(nn.Module):
#     """
#     Critic Low network(没有经验回放的版本，也就是没有批处理)
#     """

#     def __init__(self, state_dim, action_dim):
#         super(CriticLow, self).__init__()

#         self.layers = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 128),
#             nn.BatchNorm1d(128),
#             nn.Dropout(p=0.2),
#             nn.ReLU(0.2),
#             nn.Linear(128, 128),
#             nn.BatchNorm1d(128),
#             nn.Dropout(p=0.2),
#             nn.ReLU(0.2),
#             nn.Linear(128, 1),
#         )

#     def forward(self, state, action):
#         return self.layers(torch.cat([state, action], 1))


class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient
    """

    def __init__(
        self,
        agent_num,
        state_dim,
        action_dim,
        actor_lr=0.01,
        critic_lr=0.01,
        gamma=0.95,
        tau=0.01,
        batch_size=64,  # 经验回放的批大小
        load_model=False,  # 是否加载模型
        load_experience=False,  # 是否加载经验回放
        knobs=None,
        continuous_knobs_num=0,
        discrete_knobs_num=0,
    ):
        self.agent_num = agent_num
        self.critic_state_dim = state_dim * agent_num  # n个agent的state
        self.critic_action_dim = action_dim * agent_num  # n个agent的action
        self.actor_action_dim = action_dim  # actor输出的维度(单个agent的action维度)
        self.actor_state_dim = state_dim  # actor输入的维度(单个agent的state维度)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.knobs = knobs
        self.continuous_knobs_num = continuous_knobs_num
        self.discrete_knobs_num = discrete_knobs_num
        self.buffer = [
            ExperienceBuffer(state_shape=(state_dim,), action_shape=(action_dim,))
            for _ in range(agent_num)
        ]

        self.init_model()
        if load_experience:
            self.load_experience()

        if load_model:
            self.load_model()
        else:
            # 初始的目标网络参数和主网络参数一致
            self.hard_update()

    def init_model(self):
        self.actors = [
            Actor(self.actor_state_dim, self.actor_action_dim)
            for _ in range(self.agent_num)
        ]
        self.actors_target = [
            Actor(self.actor_state_dim, self.actor_action_dim)
            for _ in range(self.agent_num)
        ]
        # self.actors_sample = [
        #     ActorSample(self.actor_state_dim, self.actor_action_dim)
        #     for _ in range(self.agent_num)
        # ]
        self.critics = [
            Critic(self.critic_state_dim, self.critic_action_dim)
            for _ in range(self.agent_num)
        ]
        self.critics_target = [
            Critic(self.critic_state_dim, self.critic_action_dim)
            for _ in range(self.agent_num)
        ]

        self.actor_optimizers = [
            optimizers.Adam(actor.parameters(), lr=self.actor_lr)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            optimizers.Adam(critic.parameters(), lr=self.critic_lr)
            for critic in self.critics
        ]

        Log().Debug("MADDPG: Initialized")

    def hard_update(self):
        """
        硬更新目标网络（直接复制）
        """
        for i in range(self.agent_num):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())
            # self.actors_sample[i].load_state_dict(self.actors[i].state_dict())

    def soft_update(self):
        """
        软更新目标网络
        """
        for i in range(self.agent_num):
            for param, target_param in zip(
                self.actors[i].parameters(), self.actors_target[i].parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )
            for param, target_param in zip(
                self.critics[i].parameters(), self.critics_target[i].parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )

    # def hard_sample_update(self):
    #     """
    #     硬更新sample网络
    #     """
    #     for i in range(self.agent_num):
    #         self.actors_sample[i].load_state_dict(self.actors[i].state_dict())

    def save_model(self, path="./model/maddpg/res/"):
        """
        保存main network和target network
        """
        for i in range(self.agent_num):
            torch.save(self.actors[i].state_dict(), path + f"actor_{i}.pth")
            torch.save(self.critics[i].state_dict(), path + f"critic_{i}.pth")
            torch.save(
                self.actors_target[i].state_dict(), path + f"actor_target_{i}.pth"
            )
            torch.save(
                self.critics_target[i].state_dict(), path + f"critic_target_{i}.pth"
            )
            self.buffer[i].save(path=path, filename=f"experience_{i}.npz")

        Log().Info(f"save maddpg model and experience replay to {path} successfully")

    def load_model(self, path="./model/maddpg/res/"):
        """
        加载main network和target network
        """
        for i in range(self.agent_num):
            self.actors[i].load_state_dict(torch.load(path + f"actor_{i}.pth"))
            self.critics[i].load_state_dict(torch.load(path + f"critic_{i}.pth"))
            self.actors_target[i].load_state_dict(
                torch.load(path + f"actor_target_{i}.pth")
            )
            self.critics_target[i].load_state_dict(
                torch.load(path + f"critic_target_{i}.pth")
            )

        Log().Info(f"load maddpg model from {path} successfully")

    def load_experience(self, path="./model/maddpg/res/"):
        """
        加载经验回放
        """
        for i in range(self.agent_num):
            self.buffer[i].load(path=path, filename=f"experience_{i}.npz")

        Log().Info(f"load experience replay from {path} successfully")

    def iteratively_train(self, state, action, reward, next_state):
        """
        逐个agent训练
        """
        critic_loss_arr = []
        actor_loss_arr = []
        for i in range(self.agent_num):
            critic_loss, actor_loss = self.train(i, state, action, reward, next_state)
            critic_loss_arr.append(critic_loss)
            actor_loss_arr.append(actor_loss)

        # 软更新目标网络
        self.soft_update()
        # # 硬更新sample网络
        # self.hard_sample_update()

        return critic_loss_arr, actor_loss_arr

    def train(self, i, state, action, reward, next_state):
        """
        训练: actor(policy update)和critic(value update)
        """
        # 移除noise，使得target policy和behavior policy不一致
        self.actors[i].remove_noise()
        self.actors_target[i].remove_noise()

        critic_loss = self.update_critic(i, state, action, reward, next_state)
        actor_loss = self.update_actor(i, state)

        Log().Debug(f"train agent {i} successfully")

        return critic_loss, actor_loss

    def update_critic(self, i, state, action, reward, next_state):
        """
        value update
        """
        # 预测next_aciton
        target_action = [
            self.actors_target[j](next_state[j]) for j in range(self.agent_num)
        ]
        # 计算target_q(target value)
        next_state = torch.cat(next_state, 1)
        target_action = torch.cat(target_action, 1)
        target_q = self.critics_target[i](next_state, target_action)
        target_q = reward + self.gamma * target_q

        # 计算Q
        state = torch.cat(state, 1)
        q = self.critics[i](state, action)

        # 计算critic loss
        critic_loss = F.mse_loss(q, target_q)

        # 更新critic
        self.critic_optimizers[i].zero_grad()
        critic_loss.backward()
        self.critic_optimizers[i].step()

        return critic_loss

    def update_actor(self, i, state):
        """
        policy update
        """
        # 计算actor loss
        action = [self.actors[j](state[j]) for j in range(self.agent_num)]
        action = torch.cat(action, 1)
        state = torch.cat(state, 1)
        actor_loss = -self.critics[i](state, action).mean()

        self.actor_optimizers[i].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[i].step()

        return actor_loss

    def get_action(self, state, noise_flag=True, eval_flag=False):
        """
        noise: behavior policy get action
        """

        for i in range(self.agent_num):
            if noise_flag:
                self.actors[i].sample_noise()
            else:
                self.actors[i].remove_noise()

            if eval_flag:
                self.actors[i].eval()
            else:
                self.actors[i].train()

        action = [
            self.actors[i](state[i])[0].unsqueeze(0) for i in range(self.agent_num)
        ]
        return action

    def store_experience(self, state, action, reward, next_state):
        """
        存储经验回放
        """
        for i in range(self.agent_num):
            self.buffer[i].add(state[i], action[i], reward, next_state[i])

    def mapper_knobs(self, action):
        """
        映射action到实际的参数
        """
        res_knobs = []
        for i in range(self.agent_num):
            continuous_knobs = self.process_continuous_knobs(
                action[i][:, : self.continuous_knobs_num]
            )
            discrete_knobs = self.process_discrete_knobs(
                action[i][:, self.continuous_knobs_num :]
            )
            res_knobs.append(torch.cat([continuous_knobs, discrete_knobs], 1))

        return res_knobs

    def process_continuous_knobs(self, action):
        """
        处理连续参数
        """
        scaled_action = action * 0.5 + 0.5
        for i, knob in enumerate(self.knobs["continuous"]):
            min_val = self.knobs["continuous"][knob][0]
            max_val = self.knobs["continuous"][knob][1]
            scaled_action[:, i] = scaled_action[:, i] * (max_val - min_val) + min_val

        return scaled_action

    def process_discrete_knobs(self, action):
        """
        处理离散参数
        """
        dis_action = action.clone()  # 防止修改原action
        # 基于分桶的思想将连续值映射到离散值
        for i, knob in enumerate(self.knobs["discrete"]):
            num_buckets = len(self.knobs["discrete"][knob])
            bin_width = 2.0 / num_buckets
            dis_action[:, i] = torch.clamp(
                torch.floor((dis_action[:, i] + 1) / bin_width), 0, num_buckets - 1
            )

        return dis_action

    def get_episode(self):
        """
        获取episode
        """
        return self.buffer[0].episode

    # def get_sample_action(self, state):
    #     """
    #     使用噪声sample网络获取action
    #     """
    #     action = [self.actors_sample[i](state[i]) for i in range(self.agent_num)]
    #     return action
