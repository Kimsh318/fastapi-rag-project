import os
pid_file_path = './gunicorn_log/pid_file.txt'

def on_starting(server):
    # 서버 시작시 PID 파일 초기화
    if os.path.exists(pid_file_path): 
        os.remove(pid_file_path)

def post_fork(server, worker):
    pid = worker.pid

    # PID 기록    
    print(f"gunicorn worker pid info saved to [{pid_file_path}]")
    with open(pid_file_path, 'a') as f:
        print(f"gunicorn worker pid : {pid}")
        f.write(f"{pid}\n")