from utils.conn import Connnection
from environment.mysql_op import get_state, clear_binlog
from utils.log import Log

import time
import re
import torch


class TpccDDPG(Connnection):

    def __init__(self, host="192.168.182.132", dest_host="192.168.182.103", username="root", password="linyu50123"):
        super().__init__(host, username, password)
        self.dest_host = dest_host

    def run_tpcc(
        self,
        command="cd /root/tpcc-mysqld; ./run.sh ",
    ):
        """
        tpcc压测
        """
        state_before = get_state(host="192.168.182.103")

        Log().Debug("running tpcc to ndb cluster")
        command += self.dest_host
        stdin, stdout, stderr = self.client.exec_command(command)

        time.sleep(15)
        # time.sleep(30)
        state_after = get_state(host="192.168.182.103")

        output = stdout.read().decode("utf-8")
        error = stderr.read().decode("utf-8")
        Log().Debug("run tpcc successfully")

        if output:
            external_metrics = self.parse_tpcc_result(result=output)
            # perf = self.calculate_performance(result=external_metrics)
        if error:
            Log().Error(error)

        state = self.calculate_state(
            state_before, state_after
        )
        state = self.build_tensor(state)

        # 清空binlog
        self.clear_binlogs()

        return state, external_metrics

    def parse_tpcc_result(self, result: str):
        """
        使用正则解析tpcc结果
        """
        temporal_pattern = re.compile("<TpmC>\s+(\d+\.\d+)")
        external_metrics = {"tps": 0}
        external_metrics_lines = temporal_pattern.findall(result)

        for line in external_metrics_lines:
            external_metrics["tps"] += float(line)

        Log().Info(f"external metrics: {external_metrics}")

        return external_metrics

    def calculate_performance(self, result: dict):
        """
        计算性能的expectation
        """
        return result["tps"] * 1

    def calculate_state(self, state_before, state_after):
        """
        根据压测前与压测后的state, 计算两个node的avg(state)
        """
        state = self.calculate_avg_state(state_before, state_after)
        return state

    def calculate_avg_state(self, state_before, state_after):
        """
        计算avg(state)
        """
        state = {}
        for i in range(len(state_before)):
            state[state_before[i][0]] = (state_before[i][1] + state_after[i][1]) / 2
        return state

    def calculate_reward(self, perf_start, perf_now):
        """
        计算reward, 与初始性能相减计算增减比值, 参考DDPG论文
        """
        result = {}
        result["tps"] = float(perf_now["tps"] - perf_start["tps"]) / perf_start["tps"]

        reward = self.calculate_performance(result=result)

        if reward > 0:
            reward *= 10  # 提高正奖励, 加快收敛, 但是可能会陷入局部最优

        Log().Info(f"reward: {reward}, result: {result}")

        return reward

    def build_tensor(self, state):
        """
        将state转为tensor
        """
        state = torch.tensor(list(state.values()), dtype=torch.float32).unsqueeze(0)
        return state

    def clear_binlogs(self):
        clear_binlog(host="192.168.182.103")
        clear_binlog(host="192.168.182.104")
