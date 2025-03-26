from environment.mysql_op import get_state
from environment.knobs import init_knobs, get_knobs_template
from model.maddpg.maddpg import MADDPG

from flask import Flask
import torch
import json


def get_knobs():
    return init_knobs()


def init_model():
    mysqld_knobs, continuous_knobs_num, discrete_knobs_num = get_knobs()
    actor_lr = 0.01
    critic_lr = 0.01
    gamma = 0.95
    tau = 0.01
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

    return maddpg


def get_states(hosts: list):
    res = []
    for host in hosts:
        state = get_state(host=host)

        res_state = {}
        for i in range(len(state)):
            res_state[state[i][0]] = state[i][1]

        state = torch.tensor(list(res_state.values()), dtype=torch.float32).unsqueeze(0)
        res.append(state)

    return res


def get_action(maddpg, state, noise_flag=True):
    """
    获取action并应用到所有agent
    state: 所有agent的state
    """
    state[0] = state[0].expand(2, -1)
    state[1] = state[1].expand(2, -1)
    action = maddpg.get_action(state=state, noise_flag=noise_flag)
    knobs = maddpg.mapper_knobs(action)

    return knobs


# 创建一个 Flask 应用程序实例
app = Flask(__name__)


# 定义路由和视图函数
@app.route("/tuning")
def onlineTuning():
    hosts = ["192.168.182.103", "192.168.182.104"]
    maddpg = init_model()

    states = get_states(hosts=hosts)
    knobs = get_action(maddpg=maddpg, state=states, noise_flag=False)

    result = []
    for i in range(len(hosts)):
        knob = get_knobs_template(knobs=knobs[i], default=False)
        data = {"host": hosts[i], "knob": knob}
        result.append(data)

    json_data = json.dumps(result)
    return json_data


# 如果直接运行该脚本，则启动 Flask 应用程序
if __name__ == "__main__":
    app.run(debug=True)
