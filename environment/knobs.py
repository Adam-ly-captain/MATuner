import numpy as np
# import torch


def get_knobs_template(knobs, default=False):
    mysqld_template = """[mysqld]
basedir=/usr/local/mysql-cluster
datadir=/usr/local/mysql-cluster/data
socket=/usr/local/mysql-cluster/sock/mysql.sock
user=hadoop
symbolic-links=0
ndbcluster
ndb-connectstring=192.168.182.102
port=3306

"""

    other_template = """
[mysql_cluster]
ndb-connectstring=192.168.182.102

[ndbd]
ndb-connectstring=192.168.182.102

[mysqld_safe]
log-error=/usr/local/mysql-cluster/logs/mysql.pid
pid-file=/usr/local/mysql-cluster/logs/error.log

[mysql]
socket=/usr/local/mysql-cluster/sock/mysql.sock
"""
    if not default:
        mysqld_knobs, con_num, dis_num = init_knobs()

        i = 0
        for knob_type in mysqld_knobs:
            for knob_name, knob_value in mysqld_knobs[knob_type].items():
                mysqld_template += f"{knob_name}={int(knobs[0][i])}\n"
                i += 1

    return mysqld_template + other_template


def get_random_knobs():
    mysqld_knobs, con_num, dis_num = init_knobs()
    knobs = []
    for knob_type in mysqld_knobs:
        for knob_name, knob_value in mysqld_knobs[knob_type].items():
            if knob_type == "continuous":
                knobs.append(np.random.uniform(knob_value[0], knob_value[1]))
            else:
                knobs.append(np.random.choice(knob_value))
    return knobs


def init_knobs():
    knob_value_type = ["continuous", "discrete"]
    mysqld_knobs = {
        knob_value_type[0]: {  # [min, max]
            "innodb_buffer_pool_size": [
                1048576,
                3221225472,
            ],  # 1MB(测试时是1GB)-3G 物理内存的70%左右 影响性能最大的参数, 缓存池
            "max_connections": [129, 500],  # 最大连接数, defalt: 151  128+1, 不能低于128, 否则压测的时候因为负载均衡就会超出连接池大小无法连接
            "back_log": [100, 300],  # 连接请求队列的最大长度
            "long_query_time": [1, 20],  # 查询超过多少秒记录到慢查询日志
            "ndb_batch_size": [16384, 65536],  # ndb的批处理大小
            # "tmp_table_size": [16777216, 268435456],  # 16MB-256MB 临时表大小
            "tmp_table_size": [1024, 268435456],  # 1B-256MB 临时表大小
            # "max_heap_table_size": [16777216, 268435456],  # 16MB-256MB 最大堆表大小
            "max_heap_table_size": [16384, 268435456],  # 1B-256MB 最大堆表大小
            "table_open_cache": [1, 10000],  # 表缓存
            # "innodb_log_file_size": [
            #     536870912,
            #     2097152,  
            # ],  # 512MB-1GB 日志文件大小 可能需要删除日志文件才能生效，不删不生效，所以不一定在这里修改有用（日志太大了，不设置）
            # "innodb_log_files_in_group": [2, 10],  # 日志文件组的数量
            "innodb_flush_log_at_trx_commit": [
                0,
                2,
            ],  # 控制事务提交时日志写入磁盘的策略
            "innodb_io_capacity": [100, 2000],  # 控制InnoDB的I/O容量, 本机为固态硬盘, 所以上限为1000
            "innodb_read_io_threads": [1, 64],  # InnoDB的读线程数
            "innodb_write_io_threads": [1, 64],  # InnoDB的写线程数
            "innodb_thread_concurrency": [0, 64],  # 控制InnoDB的线程并发度
            "innodb_purge_threads": [1, 32],  # InnoDB的清理线程数
            "innodb_buffer_pool_instances": [1, 64],  # InnoDB缓冲池实例数
            "innodb_lru_scan_depth": [100, 4000],  # InnoDB LRU链表扫描深度
            "innodb_lock_wait_timeout": [1, 100],  # InnoDB事务锁等待超时时间
            "innodb_read_ahead_threshold": [0, 64], # 顺序读取时预读取页的数量
            "innodb_sync_array_size": [1, 1024],  # 控制InnoDB刷新数据和日志文件的方法
            "table_open_cache_instances": [1, 64],  # 表缓存实例数
            "thread_cache_size": [1, 1000],  # 线程缓存大小
            # "metadata_locks_hash_instances": [1, 8],  # 元数据锁哈希实例数 8.0.13后被废弃
            "innodb_spin_wait_delay": [1, 50],  # 控制InnoDB自旋等待的延迟
            "innodb_page_cleaners": [1, 16],  # InnoDB的页清理线程数
            "innodb_adaptive_hash_index_parts": [1, 128],  # 控制InnoDB自适应哈希索引的部分数
        },
        knob_value_type[1]: {  # 离散值
            "innodb_flush_method": [0, 1, 4],  # 控制InnoDB刷新数据和日志文件的方法
            "innodb_flush_neighbors": [0, 1],  # 控制InnoDB刷新数据时是否预读相邻页
            "innodb_adaptive_hash_index": [0, 1],  # 控制InnoDB是否使用自适应哈希索引
            "innodb_stats_persistent": [0, 1],  # 控制InnoDB是否持久化统计信息
            "innodb_deadlock_detect": [0, 1],  # 控制InnoDB是否检测死锁
            "innodb_random_read_ahead": [0, 1],  # 控制InnoDB是否随机读取时预读取页
        },
    }

    # 获取knobs参数数量
    continuous_knobs_num = 0
    discrete_knobs_num = 0
    for knob_type in mysqld_knobs:
        for knob_name, knob_value in mysqld_knobs[knob_type].items():
            if knob_type == knob_value_type[0]:
                continuous_knobs_num += 1
            else:
                discrete_knobs_num += 1

    return mysqld_knobs, continuous_knobs_num, discrete_knobs_num



# knobs1 = get_random_knobs()
# knobs2 = get_random_knobs()

# print(get_knobs_template(torch.tensor(knobs1).unsqueeze(0)))
