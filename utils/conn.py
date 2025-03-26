from utils.log import Log

import paramiko


class Connnection:

    def __init__(self, host, username, password):
        """
        连接到linux远程服务器
        """
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, username=username, password=password)
        self.client = client
        self.host = host
        Log().Debug(f"Connect to {host} Successfully")
        
    def execute_commands(self, commands, is_output=True):
        """
        执行多个linux命令
        """
        for command in commands:
            stdin, stdout, stderr = self.client.exec_command(command)

            # 获取命令输出
            output = stdout.read().decode("utf-8")
            error = stderr.read().decode("utf-8")

            # 输出结果
            if output and is_output:
                print(f"Command: {command}\nOutput:")
                print(output)
            if error and is_output:
                print(f"Command: {command}\nError:")
                print(error)    
    
      
    def close(self):
        """
        关闭连接
        """
        self.client.close()
        Log().Debug(f"Close connection Successfully from {self.host}")
