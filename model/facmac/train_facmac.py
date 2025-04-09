from components.episode_buffer import ReplayBuffer
from utils.logging import get_logger, Logger
from utils.timehelper import time_left, time_str
from config.config import get_args
from controllers.mac import CQMixMAC
from learners.learner import FACMACLearner
from runners.runner import MysqlRunner
from envs.mysql import get_knobs
from envs.sysbench import Sysbench
from envs.mysql_op import ConnNDB, clear_binlog

import torch
import time
import os
import pandas as pd


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

    return manager, [conn1, conn2], s
  
  
def clear_binlogs():
    clear_binlog(host="192.168.182.103")
    clear_binlog(host="192.168.182.104")
    

def close_env():
    """
    关闭environment
    """
    manager.close()
    for conn in connections:
        conn.close()
    s.close()


def load_model(args):
  if os.path.exists(args.local_results_path + "/" + "models" + "/" + args.unique_token):
      model_index = max([int(d) for d in os.listdir(args.local_results_path + "/" + "models" + "/" + args.unique_token)])
      learner.load_models(os.path.join(args.local_results_path, "models", args.unique_token, str(model_index)))
      
      
if __name__ == "__main__":
  all_start_time = time.time()
  
  all_sysbench_time = 0
  all_restart_time = 0
  all_recommend_time = 0
  all_train_time = 0
  
  manager, connections, s = init_env()

  logger = Logger(get_logger())
  args = get_args()
  metrics_logger = {}
  res = pd.read_excel('./results/excel/res.xlsx')

  groups = {
      "agents": args.n_agents
  }

  scheme = {
      "state": {"vshape": args.state_shape},
      "obs": {"vshape": args.obs_shape, "group": "agents"},
      "actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.float32},
      "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
      "reward": {"vshape": (1,)},
      "terminated": {"vshape": (1,), "dtype": torch.uint8},
  }

  preprocess = {}
  
  args = get_knobs(args=args)

  buffer = ReplayBuffer(scheme, groups, args.buffer_size, args.env_args['episode_limit'] + 1 if args.runner_scope == "episodic" else 2,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

  mac = CQMixMAC(scheme=buffer.scheme, groups=groups, args=args)

  mysqlRunner = MysqlRunner(args=args, metrics_logger=metrics_logger)
  mysqlRunner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

  learner = FACMACLearner(mac=mac, scheme=scheme, args=args, logger=logger, metrics_logger=metrics_logger)
  if args.use_cuda:
      learner.cuda()
      
  if args.load_model:
      load_model(args)
      logger.console_logger.info("Loading model from {}".format(args.local_results_path + "/" + "models" + "/" + args.unique_token))

      
  # start training
  episode = 0
  last_test_T = - args.test_interval - 1
  last_log_T = 0
  model_save_time = 0
  isFirstRun = True

  start_time = time.time()
  last_time = start_time
  start_log = False

  logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
  while mysqlRunner.t_env < args.t_max:

      # Run for a whole episode at a time
      if getattr(args, "runner_scope", "episodic") == "episodic":
          episode_batch, recommend_time, restart_time, sysbench_time, perf_start, perf_next = mysqlRunner.run(
              test_mode=False, 
              learner=learner, 
              s=s, 
              manager=manager, 
              connections=connections,
              isFirst=isFirstRun
          )
          all_recommend_time += recommend_time
          all_restart_time += restart_time
          all_sysbench_time += sysbench_time
          
          buffer.insert_episode_batch(episode_batch)

          if buffer.can_sample(args.batch_size) and (buffer.episodes_in_buffer > getattr(args, "buffer_warmup", 0)):
              episode_sample = buffer.sample(args.batch_size)

              # Truncate batch to only filled timesteps
              max_ep_t = episode_sample.max_t_filled()
              episode_sample = episode_sample[:, :max_ep_t]

              if episode_sample.device != args.device:
                  episode_sample.to(args.device)

              train_start_time = time.time()
              learner.train(episode_sample, mysqlRunner.t_env, episode)
              train_stop_time = time.time()
              all_train_time += train_stop_time - train_start_time
              
              start_log = True
      elif getattr(args, "runner_scope", "episode") == "transition":
          mysqlRunner.run(test_mode=False,
                      buffer=buffer,
                      learner=learner,
                      episode=episode,
                      s=s,
                      manager=manager,
                      connections=connections)
      else:
          raise Exception("Undefined runner scope!")

      # Execute test runs once in a while
      n_test_runs = max(1, args.test_nepisode // mysqlRunner.batch_size)
      if (mysqlRunner.t_env - last_test_T) / args.test_interval >= 1.0:

          logger.console_logger.info("t_env: {} / {}".format(mysqlRunner.t_env, args.t_max))
          logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
              time_left(last_time, last_test_T, mysqlRunner.t_env, args.t_max), time_str(time.time() - start_time)))
          last_time = time.time()

          last_test_T = mysqlRunner.t_env
          if getattr(args, "testing_on", True):
              for _ in range(n_test_runs):
                  if getattr(args, "runner_scope", "episodic") == "episodic":
                      mysqlRunner.run(test_mode=True, 
                                      learner=learner, 
                                      s=s, 
                                      manager=manager, 
                                      connections=connections)
                  elif getattr(args, "runner_scope", "episode") == "transition":
                      mysqlRunner.run(test_mode=True,
                                  buffer = buffer,
                                  learner = learner,
                                  episode = episode,
                                  s = s,
                                  manager = manager,
                                  connections = connections)
                  else:
                      raise Exception("Undefined runner scope!")

      if args.save_model and (mysqlRunner.t_env - model_save_time >= args.save_model_interval):
          model_save_time = mysqlRunner.t_env
          save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(mysqlRunner.t_env))
          #"results/models/{}".format(unique_token)
          os.makedirs(save_path, exist_ok=True)
          logger.console_logger.info("Saving models to {}".format(save_path))

          # learner should handle saving/loading -- delegate actor save/load to mac,
          # use appropriate filenames to do critics, optimizer states
          # learner.save_models(save_path, args.unique_token, model_save_time)
          
          all_stop_time = time.time()
          logger.console_logger.info("All recommend time: {}".format(all_recommend_time))
          logger.console_logger.info("All restart time: {}".format(all_restart_time))
          logger.console_logger.info("All sysbench time: {}".format(all_sysbench_time))
          logger.console_logger.info("All train time: {}".format(all_train_time))
          logger.console_logger.info("All time: {}".format(all_stop_time - all_start_time))
          
          learner.save_models(save_path)

      metrics_logger["episode"] = episode
      metrics_logger['buffersize'] = buffer.episodes_in_buffer
      episode += args.batch_size_run

      if (mysqlRunner.t_env - last_log_T) >= args.log_interval:
          logger.log_stat("episode", episode, mysqlRunner.t_env)
          logger.print_recent_stats()
          last_log_T = mysqlRunner.t_env
      
      if start_log:
          res = pd.concat([res, pd.DataFrame({
              'episode': [metrics_logger['episode']],
              't' : [metrics_logger['t']],
              'buffersize': [metrics_logger['buffersize']],
              'criticloss': [metrics_logger['criticloss']],
              'actorloss': [metrics_logger['actorloss']],
              'reward': [metrics_logger['reward']],
              "perf_tps": [perf_next["tps"]],
              "perf_qps": [perf_next["qps"]],
              "perf_lat": [perf_next["lat"]],
              "perf_start_tps": [perf_start["tps"]],
              "perf_start_qps": [perf_start["qps"]],
              "perf_start_lat": [perf_start["lat"]],
          })])
          res.to_excel('./results/excel/res.xlsx', index=False)
      
      logger.console_logger.info("t_env: {} / {}".format(mysqlRunner.t_env, args.t_max))
      metrics_logger.clear()
      isFirstRun = False
      
  close_env()
  
      