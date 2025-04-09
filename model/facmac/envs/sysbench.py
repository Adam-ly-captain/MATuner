from utils.conn import Connnection
from envs.mysql_op import get_state, clear_binlog

import time
import re
import torch


class Sysbench(Connnection):

    def __init__(self, host="192.168.182.132", username="root", password="linyu50123"):
        super().__init__(host, username, password)

    def run_sysbench(
        self,
        command="cd /root/ndb-test; ./run.sh",
    ):
        """
        sysbench压测
        """
        state1_before = get_state(host="192.168.182.103")
        state2_before = get_state(host="192.168.182.104")

        stdin, stdout, stderr = self.client.exec_command(command)

        # time.sleep(5)
        time.sleep(30)
        state1_after = get_state(host="192.168.182.103")
        state2_after = get_state(host="192.168.182.104")

        output = stdout.read().decode("utf-8")
        error = stderr.read().decode("utf-8")

        if output:
            external_metrics = self.parse_sysbench_result(result=output)
            # perf = self.calculate_performance(result=external_metrics)
        if error:
            print(f"Error: {error}")

        state1, state2 = self.calculate_state(
            state1_before, state1_after, state2_before, state2_after
        )
        state1, state2 = self.build_tensor(state1, state2)

        # 清空binlog
        self.clear_binlogs()

        return state1, state2, external_metrics

    def parse_sysbench_result(self, result: str):
        """
        使用正则解析sysbench结果, 计算reward参考DDPG论文
        """
        temporal_pattern = re.compile(
            "tps: (\d+\.\d+) qps: (\d+\.\d+) \(r/w/o: (\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)\)"
            " lat \(ms,95%\): (\d+\.\d+) err/s: (\d+\.\d+) reconn/s: (\d+\.\d+)"
        )

        external_metrics = {"tps": 0, "qps": 0, "lat": 0}
        external_metrics_lines = temporal_pattern.findall(result)
        lines = len(external_metrics_lines[-5:])
        for line in external_metrics_lines[-5:]:
            external_metrics["tps"] += float(line[0])
            external_metrics["qps"] += float(line[1])
            external_metrics["lat"] += float(line[5])

        external_metrics["tps"] /= lines
        external_metrics["qps"] /= lines
        external_metrics["lat"] /= lines

        # Log().Info(f"external metrics: {external_metrics}")

        return external_metrics

    def calculate_performance(self, result: dict):
        """
        计算性能的expectation
        """
        return result["tps"] * 0.25 + result["qps"] * 0.25 + result["lat"] * 0.5

    def calculate_state(self, state1_before, state1_after, state2_before, state2_after):
        """
        根据压测前与压测后的state, 计算两个node的avg(state)
        """
        state1 = self.calculate_avg_state(state1_before, state1_after)
        state2 = self.calculate_avg_state(state2_before, state2_after)
        return state1, state2

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
        result["qps"] = float(perf_now["qps"] - perf_start["qps"]) / perf_start["qps"]
        result["lat"] = float(perf_start["lat"] - perf_now["lat"]) / perf_start["lat"]

        reward = self.calculate_performance(result=result)

        if reward > 0:
            reward *= 10  # 提高正奖励, 加快收敛, 但是可能会陷入局部最优

        return reward

    def build_tensor(self, state1, state2):
        """
        将state1, state2转为tensor
        """
        state1 = torch.tensor(list(state1.values()), dtype=torch.float32).unsqueeze(0)
        state2 = torch.tensor(list(state2.values()), dtype=torch.float32).unsqueeze(0)

        return state1, state2

    def clear_binlogs(self):
        clear_binlog(host="192.168.182.103")
        clear_binlog(host="192.168.182.104")
