from utils.conn import Connnection
from utils.log import Log
from environment.knobs import get_knobs_template

from datetime import datetime
import mysql.connector


def get_mysql_conn(host):
    """
    host: 192.168.182.103/104
    get_state(host='192.168.182.103')
    """
    # 连接到MySQL数据库
    conn = mysql.connector.connect(
        host=host, user="hadoop", password="123456", database="test"
    )

    return conn


def get_state(host):
    """
    获取74个state metrics
    """

    conn = get_mysql_conn(host)

    # 创建游标对象
    cursor = conn.cursor()

    try:
        # 执行SQL查询
        # cursor.execute("SHOW STATUS")

        cursor.execute(
            'SELECT NAME, COUNT from information_schema.INNODB_METRICS where status="enabled" ORDER BY NAME;'
        )

        # 获取查询结果
        result = cursor.fetchall()
        return result

    except Exception as e:
        Log().Error(f"get state failed: {e}")

    finally:
        # 关闭游标和数据库连接
        cursor.close()

        close_conn(conn=conn)


def close_conn(conn):
    """
    关闭数据库连接
    """
    conn.close()


def clear_binlog(host):
    """
    清空binlog, 不定时清除磁盘会因为不断压测而爆满
    """
    # 获取当前日期和时间
    current_datetime = datetime.now()

    # 将日期和时间格式化为 'yyyy-mm-dd hh:mm:ss' 格式的字符串
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # 连接到MySQL数据库
    conn = mysql.connector.connect(
        host=host, user="hadoop", password="123456", database="test"
    )

    # 创建游标对象
    cursor = conn.cursor()

    cursor.execute(f"PURGE BINARY LOGS BEFORE '{formatted_datetime}'")
    
    cursor.close()
    conn.close()   

    Log().Debug(f"purge binary logs from {host} successfully")


class ConnNDB(Connnection):
    """
    远程连接NDB集群
    client连接需要手动调用close方法关闭
    """

    def __init__(self, host, username="hadoop", password="linyu50123"):
        super().__init__(host, username, password)

    def modify_knobs(
        self,
        knobs,
        remote_path="/etc/my.cnf",
    ):
        """
        修改SQL节点的my.cnf
        """

        # 打开远程文件并写入内容
        sftp = self.client.open_sftp()
        with sftp.file(remote_path, "w") as file:
            file.write(knobs)
            Log().Debug(f"modify knobs successfully to {remote_path} from {self.host}")

        # 关闭SFTP会话和SSH连接
        sftp.close()

    def restart_mysql_cluster(self):
        """
        执行restart.sh重启NDB集群
        """
        Log().Debug(f"restart mysql cluster from {self.host}")

        self.execute_commands(
            ["cd /usr/local/mysql/mysql-cluster; ./restart.sh"], is_output=False
        )  # 真正开始训练模型的时候改为False，减少终端的输出

        Log().Debug(f"restart mysql cluster successfully from {self.host}")

    def restart_mysqld_node(self):
        """
        重启mysqld节点
        """
        Log().Debug(f"restart mysqld node from {self.host}")

        self.execute_commands(
            ["cd /home/hadoop/bin; ./restart-mysqld"], is_output=False
        )

        Log().Debug(f"restart mysqld node successfully from {self.host}")

    def apply_knobs(self, knobs, remote_path="/etc/my.cnf", default=False):
        """
        获取my.cnf配置模板, 应用my.cnf配置
        """
        mysqld_knobs = get_knobs_template(knobs=knobs, default=default)
        self.modify_knobs(knobs=mysqld_knobs, remote_path=remote_path)

    def clear_redo_log(self):
        """
        清空redo log
        """
        self.execute_commands(["rm -f /usr/local/mysql-cluster/data/#innodb_redo/*"])
        Log().Debug(f"clear redo log from {self.host} successfully")


# Usage:
# n = ConnNDB(host='192.168.182.132', username='root', password='linyu50123')
# n.purge_binary_logs('yyyy-mm-dd hh:mm:ss')
# n.close()


# n = ConnNDB(host='192.168.182.132', username='root', password='linyu50123')
# n1 = ConnNDB(host='192.168.182.132', username='root', password='linyu50123')
# n.execute_commands(['ls'])
# n1.execute_commands(['ls -l'])
# n1.modify_knobs(knobs='12345', remote_path='/root/b.txt')
# n.close()
# n1.close()
