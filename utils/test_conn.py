import paramiko

def execute_commands(hostname, username, password, commands):
    # 创建SSH客户端对象
    client = paramiko.SSHClient()
    # 自动添加主机密钥
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # 连接到远程主机
        client.connect(hostname, username=username, password=password)
        
        # 执行多个命令
        for command in commands:
            stdin, stdout, stderr = client.exec_command(command)
            
            # 获取命令输出
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            
            # 输出结果
            if output:
                print(f"Command: {command}\nOutput:")
                print(output)
            if error:
                print(f"Command: {command}\nError:")
                print(error)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 关闭SSH连接
        client.close()

# 虚拟机信息
hostname = '192.168.182.132'
username = 'root'
password = 'linyu50123'

# 要执行的多个命令
# commands = ['cd /usr/local/mysql/mysql-cluster; ./restart.sh']
commands = ['ls']

# 执行命令
execute_commands(hostname, username, password, commands)
