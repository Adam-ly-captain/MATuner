from environment.mysql_op import ConnNDB, clear_binlog
from environment.sysbench import Sysbench
from environment.tpcc import Tpcc
from environment.knobs import init_knobs, get_random_knobs
from model.maddpg_src.maddpg import MADDPG
from utils.log import Log

import numpy as np
import pandas as pd
import torch


def init_env():
    """
    初始化environment
    """

    # 初始化集群连接
    manager = ConnNDB(host="192.168.182.102")
    conn1 = ConnNDB(host="192.168.182.103")
    conn2 = ConnNDB(host="192.168.182.104")

    # mysql_conn1 = get_mysql_conn(host="192.168.182.103")
    # mysql_conn2 = get_mysql_conn(host="192.168.182.104")

    # 初始化sysbench
    s = Sysbench()
    t = Tpcc()
    # s.run_sysbench()
    clear_binlogs()

    Log().Debug("Init environment successfully")
    return manager, [conn1, conn2], s, t


def clear_binlogs():
    clear_binlog(host="192.168.182.103")
    clear_binlog(host="192.168.182.104")


def get_knobs():
    return init_knobs()


def close_env():
    """
    关闭environment
    """
    manager.close()
    for conn in connections:
        conn.close()
    s.close()
    t.close()

    # for conn in mysql_conns:
    #     close_conn(conn)

    Log().Debug("Close environment successfully")


def get_action(state, noise_flag=False, default=False):
    """
    获取action并应用到所有agent
    state: 所有agent的state
    """
    if not default:
        state1 = state[0].expand(2, -1)
        state2 = state[1].expand(2, -1)
        new_state = [state1, state2]
        action = maddpg.get_action(state=new_state, noise_flag=noise_flag)
        # action = maddpg.get_sample_action(state=state)
        knobs = maddpg.mapper_knobs(action)

        for i in range(len(connections)):
            connections[i].apply_knobs(knobs[i])
    else:
        for i in range(len(connections)):
            connections[i].apply_knobs(knobs=None, default=True)

        action = None

    # 重启集群
    manager.restart_mysql_cluster()
    return action


def get_random_action():
    knobs1 = get_random_knobs()
    knobs2 = get_random_knobs()
    knobs1 = torch.tensor(knobs1).unsqueeze(0)
    knobs2 = torch.tensor(knobs2).unsqueeze(0)
    knobs = [knobs1, knobs2]
    for i in range(len(connections)):
        connections[i].apply_knobs(knobs=knobs[i])

    manager.restart_mysql_cluster()


def get_random_sample(noise_flag=False):
    get_random_action()
    state1, state2, perf_start = s.run_sysbench()  # avg(state1), avg(state2), perf
    action = get_action(state=[state1, state2], noise_flag=noise_flag)
    next_state1, next_state2, perf_next = s.run_sysbench()
    reward = s.calculate_reward(perf_start=perf_start, perf_now=perf_next)
    next_state = [next_state1, next_state2]
    state = [state1, state2]
    return state, action, reward, next_state, perf_start, perf_next


def get_random_sample_tpcc(noise_flag=False):
    get_random_action()
    state1, state2, perf_start = t.run_tpcc()  # avg(state1), avg(state2), perf
    action = get_action(state=[state1, state2], noise_flag=noise_flag)
    next_state1, next_state2, perf_next = t.run_tpcc()
    reward = t.calculate_reward(perf_start=perf_start, perf_now=perf_next)
    next_state = [next_state1, next_state2]
    state = [state1, state2]
    return state, action, reward, next_state, perf_start, perf_next


def get_first_sample(noise_flag=False):
    get_action(state=None, noise_flag=False, default=True)
    state1, state2, perf_start = s.run_sysbench()  # avg(state1), avg(state2), perf
    action = get_action(state=[state1, state2], noise_flag=noise_flag)
    next_state1, next_state2, perf_next = s.run_sysbench()
    reward = s.calculate_reward(perf_start=perf_start, perf_now=perf_next)
    next_state = [next_state1, next_state2]
    state = [state1, state2]
    return state, action, reward, next_state, perf_start, perf_next


def get_first_sample_tpcc(noise_flag=False):
    get_action(state=None, noise_flag=False, default=True)
    state1, state2, perf_start = t.run_tpcc()  # avg(state1), avg(state2), perf
    action = get_action(state=[state1, state2], noise_flag=noise_flag)
    next_state1, next_state2, perf_next = t.run_tpcc()
    reward = t.calculate_reward(perf_start=perf_start, perf_now=perf_next)
    next_state = [next_state1, next_state2]
    state = [state1, state2]
    return state, action, reward, next_state, perf_start, perf_next


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True) # 开启异常检测

    data = pd.read_excel("./model/maddpg_src/res/eval_read_only.xlsx")

    manager, connections, s, t = init_env()
    mysqld_knobs, continuous_knobs_num, discrete_knobs_num = get_knobs()

    episode = 1
    actor_lr = 0.01
    critic_lr = 0.01
    gamma = 0.95
    tau = 0.01
    buffer_size = 32
    agent_num = 2

    maddpg = MADDPG(
        agent_num=agent_num,
        state_dim=74,  # 74个internal metrics
        action_dim=continuous_knobs_num + discrete_knobs_num,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        tau=tau,
        knobs=mysqld_knobs,
        continuous_knobs_num=continuous_knobs_num,
        discrete_knobs_num=discrete_knobs_num,
        load_model=True,
        load_experience=True,
    )

    if data.size == 0:
        real_episode = 0
    else:
        real_episode = data["episode"].max() + 1

    for i in range(episode):
        state, action, reward, next_state, perf_start, perf_next = get_first_sample(
            noise_flag=False
        )

        data = pd.concat(
            [
                data,
                pd.DataFrame(
                    {
                        "episode": [real_episode],
                        "default_tps": [perf_start["tps"]],
                        "default_qps": [perf_start["qps"]],
                        "default_lat": [perf_start["lat"]],
                        "tune_tps": [perf_next["tps"]],
                        "tune_qps": [perf_next["qps"]],
                        "tune_lat": [perf_next["lat"]],
                    }
                ),
            ]
        )

        real_episode += 1

    data.to_excel("./model/maddpg_src/res/eval_read_only.xlsx", index=False)

    # state, action, reward, next_state, perf_start = get_first_sample(noise_flag=False)

    # get_random_sample()

    # get_first_sample_tpcc()
