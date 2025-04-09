from environment.mysql_op import ConnNDB, clear_binlog
from environment.sysbench import Sysbench
from environment.knobs import init_knobs
from model.maddpg.maddpg import MADDPG
from utils.log import Log

import numpy as np
import pandas as pd
import torch
import time


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
    # s.run_sysbench()
    clear_binlogs()

    Log().Debug("Init environment successfully")
    return manager, [conn1, conn2], s


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

    # for conn in mysql_conns:
    #     close_conn(conn)

    Log().Debug("Close environment successfully")


def get_action(state, noise_flag=True):
    """
    获取action并应用到所有agent
    state: 所有agent的state
    """
    state1 = state[0].expand(2, -1)
    state2 = state[1].expand(2, -1)
    new_state = [state1, state2]

    start_time = time.time()
    action = maddpg.get_action(state=new_state, noise_flag=noise_flag)
    # action = maddpg.get_sample_action(state=state)
    knobs = maddpg.mapper_knobs(action)
    for i in range(len(connections)):
        connections[i].apply_knobs(knobs[i])

    stop_time = time.time()
    recommend_time = stop_time - start_time

    # 重启集群
    start_time = time.time()
    manager.restart_mysql_cluster()
    stop_time = time.time()

    restart_time = stop_time - start_time

    return action, recommend_time, restart_time


def get_first_sample():
    all_sysbench_time = 0

    sysbench_start_time = time.time()
    state1, state2, perf_start = s.run_sysbench()  # avg(state1), avg(state2), perf
    sysbench_stop_time = time.time()
    all_sysbench_time += sysbench_stop_time - sysbench_start_time

    action, recommend_time, restart_time = get_action(state=[state1, state2])

    sysbench_start_time = time.time()
    next_state1, next_state2, perf_next = s.run_sysbench()
    sysbench_stop_time = time.time()
    all_sysbench_time += sysbench_stop_time - sysbench_start_time

    reward = s.calculate_reward(perf_start=perf_start, perf_now=perf_next)
    next_state = [next_state1, next_state2]
    state = [state1, state2]
    return (
        state,
        action,
        reward,
        next_state,
        perf_start,
        recommend_time,
        restart_time,
        all_sysbench_time,
    )


def get_sample(state, perf_start):
    action, recommend_time, restart_time = get_action(state=state)

    sysbench_start_time = time.time()
    state1, state2, perf = s.run_sysbench()  # avg(state1), avg(state2), perf
    sysbench_stop_time = time.time()

    reward = s.calculate_reward(perf_start=perf_start, perf_now=perf)
    next_state = [state1, state2]
    return (
        action,
        reward,
        next_state,
        perf,
        recommend_time,
        restart_time,
        sysbench_stop_time - sysbench_start_time,
    )


# def clear_redo_log():
#     """
#     清空redo log
#     """
#     for conn in connections:
#         conn.clear_redo_log()


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True) # 开启异常检测

    all_start_time = time.time()

    manager, connections, s = init_env()
    mysqld_knobs, continuous_knobs_num, discrete_knobs_num = get_knobs()
    episode_data = pd.read_excel("./model/maddpg/res/res_tps0.25_lat0.5.xlsx")

    episode = 5
    max_episode_length = 50  # 因为我的perf_start的计算是从一个episode的开始才定义，所以尽量把max_episode_length设置大一点
    actor_lr = 0.01
    critic_lr = 0.01
    gamma = 0.95
    tau = 0.01
    buffer_size = 128
    agent_num = 2

    maddpg = MADDPG(
        agent_num=agent_num,
        state_dim=74,  # 74个internal metrics
        action_dim=continuous_knobs_num + discrete_knobs_num,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        tau=tau,
        batch_size=buffer_size,
        knobs=mysqld_knobs,
        continuous_knobs_num=continuous_knobs_num,
        discrete_knobs_num=discrete_knobs_num,
        load_model=True,
        load_experience=True,
    )

    all_recommend_time = 0
    all_restart_time = 0
    all_backward_time = 0
    all_sysbench_test_time = 0
    
    if episode_data.size == 0:
        real_episode = 0
    else:
        real_episode = episode_data["episode"].max() + 1
    
    for i in range(episode):
        t = 0
        all_negative_reward = 0

        (
            state,
            action,
            reward,
            next_state,
            perf_start,
            recommend_time,
            restart_time,
            sysbench_time,
        ) = get_first_sample()

        all_recommend_time += recommend_time
        all_restart_time += restart_time
        all_sysbench_test_time += sysbench_time

        maddpg.store_experience(state, action, reward, next_state)
        while (
            t < max_episode_length and all_negative_reward > -12.5
        ):  # -12.5 计算大概有一半以上(25)的reward超过0.1
            state = next_state
            (
                action,
                reward,
                next_state,
                perf,
                recommend_time,
                restart_time,
                sysbench_time,
            ) = get_sample(state=next_state, perf_start=perf_start)

            all_recommend_time += recommend_time
            all_restart_time += restart_time
            all_sysbench_test_time += sysbench_time

            if maddpg.buffer[0].size > buffer_size:
                state_arr = []
                action_arr = []
                reward_arr = []
                next_state_arr = []
                for buffer in maddpg.buffer:
                    state1, action1, reward1, next_state1 = buffer.sample(
                        batch_size=buffer_size
                    )
                    state_arr.append(state1)
                    action_arr.append(action1)
                    reward_arr = reward1
                    next_state_arr.append(next_state1)
                action_arr = torch.cat(action_arr, 1)
                start_time = time.time()
                critic_loss_arr, actor_loss_arr = maddpg.iteratively_train(
                    state_arr, action_arr, reward_arr, next_state_arr
                )
                stop_time = time.time()

                backward_time = stop_time - start_time
                all_backward_time += backward_time

                critic_loss_arr = torch.tensor(critic_loss_arr)
                actor_loss_arr = torch.tensor(actor_loss_arr)

                Log().Info(
                    f"Episode: {real_episode}, Step: {t}, Reward: {reward}, Buffersize: {maddpg.buffer[0].size}, Critic Loss: {torch.mean(critic_loss_arr).item()}, Actor Loss: {torch.mean(actor_loss_arr).item()}"
                )

                episode_data = pd.concat(
                    [
                        episode_data,
                        pd.DataFrame(
                            {
                                "episode": [real_episode],
                                "t": [t],
                                "reward": [reward],
                                "buffersize": [maddpg.buffer[0].size],
                                "criticloss": [torch.mean(critic_loss_arr).item()],
                                "actorloss": [torch.mean(actor_loss_arr).item()],
                                "perf_tps": [perf["tps"]],
                                "perf_qps": [perf["qps"]],
                                "perf_lat": [perf["lat"]],
                                "perf_start_tps": [perf_start["tps"]],
                                "perf_start_qps": [perf_start["qps"]],
                                "perf_start_lat": [perf_start["lat"]],
                            }
                        ),
                    ]
                )

            maddpg.store_experience(state, action, reward, next_state)

            if reward < 0:
                all_negative_reward += reward
            t += 1

        real_episode += 1

        # 一个episode保存一次
        maddpg.save_model()

        episode_data.to_excel("./model/maddpg/res/res_tps0.25_lat0.5.xlsx", index=False)

    close_env()

    all_stop_time = time.time()

    Log().Info(
        f"execution time: {all_stop_time - all_start_time}, recommend time: {all_recommend_time}, restart time: {all_restart_time}, backward time: {all_backward_time}, sysbench test time: {all_sysbench_test_time}"
    )

