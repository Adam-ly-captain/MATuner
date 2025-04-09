from functools import partial
from components.episode_buffer import EpisodeBatch
from envs.knobs import get_random_knobs
import torch as th
import numpy as np
import time


class MysqlRunner:

    def __init__(self, args, metrics_logger=None):
        self.args = args
        self.metrics_logger = metrics_logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.episode_limit = args.env_args['episode_limit']
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.max_state = []
        self.min_state = []

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def reset(self):
        self.batch = self.new_batch()
        self.t = 0
        
    def get_state(self, s):
        state1, state2, perf_start = s.run_sysbench()  # avg(state1), avg(state2), perf
        new_state1, new_state2 = state1.clone(), state2.clone()
        
        # 按维度 min-max 归一化
        if isinstance(self.max_state, list) and len(self.max_state) == 0:
            for i in range(self.args.state_shape // self.args.n_agents):
                self.max_state.append(th.max(new_state1[0][i], new_state2[0][i]))
                self.min_state.append(th.min(new_state1[0][i], new_state2[0][i]))

            self.max_state = th.tensor(self.max_state)
            self.min_state = th.tensor(self.min_state)
            
        elif isinstance(self.max_state, th.Tensor):
            for i in range(self.args.state_shape // self.args.n_agents):
                self.max_state[i] = max(self.max_state[i].item(), new_state1[0][i].item(), new_state2[0][i].item())
                self.min_state[i] = min(self.min_state[i].item(), new_state1[0][i].item(), new_state2[0][i].item())
        
        for i in range(self.args.state_shape // self.args.n_agents):
            if self.max_state[i] == self.min_state[i]:
                new_state1[0][i] = 0
                new_state2[0][i] = 0
            else:
                new_state1[0][i] = (new_state1[0][i] - self.min_state[i]) / (self.max_state[i] - self.min_state[i])
                new_state2[0][i] = (new_state2[0][i] - self.min_state[i]) / (self.max_state[i] - self.min_state[i])
            
        state = th.cat([new_state1, new_state2], dim=1).reshape(new_state1.shape[1] + new_state2.shape[1]).detach().cpu().numpy()
        
        return state, new_state1, new_state2, perf_start

    def get_obs_random(self):
      _obs = th.tensor(get_random_knobs()).unsqueeze(0)
      for i in range(1, self.args.n_agents):
          _obs_i = th.tensor(get_random_knobs()).unsqueeze(0)
          _obs = th.cat([_obs, _obs_i], dim=0)
      
      return _obs.detach().cpu().numpy()
    
    def get_obs(self, test_mode=False):
        actions, origin_actions = self.get_actions(test_mode=test_mode)
        _obs = origin_actions.reshape(self.args.n_agents, self.args.n_actions)
        
        return _obs.detach().cpu().numpy()
      
    def get_actions(self, test_mode=False):
        if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
            actions, origin_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode,
                                              explore=(not test_mode))
        else:
            actions, origin_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
            origin_actions = th.argmax(origin_actions, dim=-1).long()
            actions = th.argmax(actions, dim=-1).long()
        
        return actions, origin_actions

    def run(self, test_mode=False, s=None, manager=None, connections=[], isFirst=True, **kwargs):
        self.reset()
        restart_time = 0
        recommend_time = 0
        sysbench_time = 0
    
        terminated = False
        self.mac.init_hidden(batch_size=self.batch_size)
        
        start_time = time.time()
        state, state1, state2, perf_start = self.get_state(s)
        stop_time = time.time()
        sysbench_time += stop_time - start_time
        
        avail_actions = np.random.normal(size=(self.args.n_actions, self.args.n_agents))  # TODO
        if isFirst:
          obs = self.get_obs_random()
        else:
          obs = self.get_obs(test_mode=test_mode)
        
        pre_transition_data = {
            "state": [state],
            "avail_actions": [avail_actions],
            "obs": [obs]
        }
        print(state)
        self.batch.update(pre_transition_data, ts=self.t)
        
        start_time = time.time()
        actions, origin_actions = self.get_actions(test_mode=test_mode)
        stop_time = time.time()
        recommend_time += stop_time - start_time
        
        for i in range(len(connections)):
            connections[i].apply_knobs(actions[0][i].reshape(1, self.args.n_actions))

        # 重启集群
        start_time = time.time()
        manager.restart_mysql_cluster()
        stop_time = time.time()
        restart_time += stop_time - start_time
        
        # get next state
        start_time = time.time()
        state, state1, state2, perf_next = self.get_state(s)
        stop_time = time.time()
        sysbench_time += stop_time - start_time
        
        reward = np.array([s.calculate_reward(perf_start=perf_start, perf_now=perf_next)])
        terminated = True
        env_info = {"episode_limit": False}

        post_transition_data = {
            "actions": origin_actions,
            "reward": [(reward,)],
            "terminated": [(terminated != env_info.get("episode_limit", False),)],
        }
        
        self.metrics_logger['reward'] = reward[0]

        self.batch.update(post_transition_data, ts=self.t)
        self.t += 1
        
        avail_actions = np.random.normal(size=(self.args.n_actions, self.args.n_agents)) # TODO
        if isFirst:
          obs = self.get_obs_random()
        else:
          obs = self.get_obs(test_mode=test_mode)
        
        last_data = {
            "state": [state],
            "avail_actions": [avail_actions],
            "obs": [obs]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
            actions, origin_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode,
                                              explore=(not test_mode))
        else:
            actions, origin_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
            origin_actions = th.argmax(origin_actions, dim=-1).long()
            actions = th.argmax(actions, dim=-1).long()

        self.batch.update({"actions": origin_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(reward[0])

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            # self._log(cur_returns, cur_stats, log_prefix)
            pass
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            # self._log(cur_returns, cur_stats, log_prefix)
            pass
            if self.args.action_selector is not None and hasattr(self.mac.action_selector, "epsilon"):
                # self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
                pass
            self.log_train_stats_t = self.t_env
        
        return self.batch, recommend_time, restart_time, sysbench_time, perf_start, perf_next

