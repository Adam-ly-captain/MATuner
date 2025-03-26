from environment.mysql_op import ConnNDB, clear_binlog
from environment.sysbench_ddpg import SysbenchDDPG
from environment.knobs import init_knobs
from model.ddpg.ddpg import DDPG
from utils.log import Log

# import numpy as np
import pandas as pd
# import torch
import time


def init_env(host):
    """
    初始化environment
    """

    # 初始化数据库节点连接
    connection = ConnNDB(host=host)

    # 初始化sysbench
    s = SysbenchDDPG(dest_host=host)

    clear_binlogs(host=host)

    Log().Debug("Init environment successfully")
    return connection, s


def clear_binlogs(host):
    clear_binlog(host=host)


def get_knobs():
    return init_knobs()


def close_env():
    """
    关闭environment
    """
    connection.close()
    s.close()

    Log().Debug("Close environment successfully")


def get_action(state, noise_flag=True):
    """
    获取action并应用到指定数据库节点
    state: 单个agent的state
    """
    new_state = state.expand(2, -1)

    start_time = time.time()
    action = ddpg.get_action(state=new_state, noise_flag=noise_flag)
    knobs = ddpg.mapper_knobs(action)
    connection.apply_knobs(knobs)
    stop_time = time.time()

    recommend_time = stop_time - start_time

    # 重启数据库节点
    start_time = time.time()
    connection.restart_mysqld_node()
    stop_time = time.time()

    restart_time = stop_time - start_time

    return action, recommend_time, restart_time


def get_first_sample():
    all_sysbench_time = 0

    sysbench_start_time = time.time()
    state, perf_start = s.run_sysbench()  # avg(state), perf
    sysbenc_stop_time = time.time()

    all_sysbench_time += sysbenc_stop_time - sysbench_start_time

    action, recommend_time, restart_time = get_action(state=state)

    sysbench_start_time = time.time()
    next_state, perf_next = s.run_sysbench()
    sysbenc_stop_time = time.time()

    all_sysbench_time += sysbenc_stop_time - sysbench_start_time

    reward = s.calculate_reward(perf_start=perf_start, perf_now=perf_next)

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
    next_state, perf = s.run_sysbench()  # avg(state), perf
    sysbench_stop_time = time.time()

    reward = s.calculate_reward(perf_start=perf_start, perf_now=perf)

    return (
        action,
        reward,
        next_state,
        perf,
        recommend_time,
        restart_time,
        sysbench_stop_time - sysbench_start_time,
    )


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True) # 开启异常检测

    exec_start_time = time.time()

    node_id = 0
    host = "192.168.182.103"
    connection, s = init_env(host=host)
    mysqld_knobs, continuous_knobs_num, discrete_knobs_num = get_knobs()
    episode_data = pd.read_excel("./model/ddpg/res/res.xlsx")

    episode = 3
    max_episode_length = 50  # 因为我的perf_start的计算是从一个episode的开始才定义，所以尽量把max_episode_length设置大一点
    actor_lr = 0.01
    critic_lr = 0.01
    gamma = 0.95
    tau = 0.01
    buffer_size = 128

    ddpg = DDPG(
        node_id=node_id,
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

        ddpg.store_experience(state, action, reward, next_state)
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

            if ddpg.buffer.size > buffer_size:
                state1, action1, reward1, next_state1 = ddpg.buffer.sample(
                    batch_size=buffer_size
                )

                time_start = time.time()
                critic_loss, actor_loss = ddpg.iteratively_train(
                    state1, action1, reward1, next_state1
                )
                time_end = time.time()

                all_backward_time += time_end - time_start

                Log().Info(
                    f"Episode: {real_episode}, Step: {t}, Reward: {reward}, Buffersize: {ddpg.buffer.size}, Critic Loss: {critic_loss}, Actor Loss: {actor_loss}"
                )

                episode_data = pd.concat(
                    [
                        episode_data,
                        pd.DataFrame(
                            {
                                "episode": [real_episode],
                                "t": [t],
                                "reward": [reward],
                                "buffersize": [ddpg.buffer.size],
                                "criticloss": critic_loss.detach().numpy(),
                                "actorloss": actor_loss.detach().numpy(),
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

            ddpg.store_experience(state, action, reward, next_state)

            if reward < 0:
                all_negative_reward += reward
            t += 1

        real_episode += 1

        # 一个episode保存一 次
        ddpg.save_model()

        episode_data.to_excel("./model/ddpg/res/res.xlsx", index=False)

    close_env()

    exec_end_time = time.time()

    Log().Info(
        f"exec time: {exec_end_time - exec_start_time}, recommend time: {all_recommend_time}, restart time: {all_restart_time}, backward time: {all_backward_time}, sysbench test time: {all_sysbench_test_time}"
    )

    # # Randomly generate data
    # state_dim = 74
    # action_dim = continuous_knobs_num + discrete_knobs_num
    # batch_size = 1

    # state = torch.randn(batch_size, state_dim).expand(2, -1)
    # action = torch.randn(batch_size, action_dim).expand(2, -1)
    # reward = torch.randn(batch_size, 1)
    # next_state = torch.randn(batch_size, state_dim).expand(2, -1)

    # # Perform iterative training
    # ddpg.iteratively_train(state, action, reward, next_state)

    # print(ddpg.mapper_knobs(ddpg.get_action(state)))
