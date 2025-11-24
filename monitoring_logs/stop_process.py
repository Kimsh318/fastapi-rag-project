import os
import subprocess

# 확인하고 싶은 싶은 포트번호
# FastAPI
## ML1 : 8000 => http://10.6.40.76:32050
## ML6 : 5000 => http://http://10.6.40.90:32018

# Gradio
## ML1 : 8002 => http://10.6.40.76:32052
port = 8000

def check_port_status(port):
    """특정 포트의 상태를 확인합니다."""
    hex_port = format(port, '04X')
    status_codes = {
        '01': 'ESTABLISHED',
        '02': 'SYN_SENT',
        '03': 'SYN_RECV',
        '04': 'FIN_WAIT1',
        '05': 'FIN_WAIT2',
        '06': 'TIME_WAIT',
        '07': 'CLOSE',
        '08': 'CLOSE_WAIT',
        '09': 'LAST_ACK',
        '0A': 'LISTEN',
        '0B': 'CLOSING'
    }

    with open('/proc/net/tcp', 'r') as f:
        for line in f.readlines()[1:]:  # 첫 줄(헤더)은 건너뜁니다
            parts = line.split()
            local_address = parts[1].split(':')[1]
            status = parts[3]
           
            if local_address.lower() == hex_port.lower():
                return status_codes.get(status, 'UNKNOWN')

    return 'NOT_FOUND'

# 사용 예시
port_status = check_port_status(port)

if port_status == 'TIME_WAIT':
    print(f"{port}번 포트는 현재 TIME_WAIT 상태에 있습니다.")
elif port_status == 'ESTABLISHED':
    print(f"{port}번 포트는 현재 사용 중(ESTABLISHED)입니다.")
elif port_status == 'NOT_FOUND':
    print(f"{port}번 포트를 사용하는 프로세스를 찾을 수 없습니다.")
else:
    print(f"{port}번 포트는 현재 {port_status} 상태에 있습니다.")



def convert_port_to_hex(port):
    """포트 번호를 16진수로 변환합니다."""
    return format(port, '04X')

def read_network_file(file, hex_port):
    """네트워크 파일을 읽고 특정 포트를 찾습니다."""
    with open(file, 'r') as f:
        for line in f.readlines()[1:]:  # 첫 줄(헤더)은 건너뜁니다
            parts = line.split()
            local_address = parts[1].split(':')[1]  # 로컬 포트 부분만 추출
            if local_address.lower() == hex_port.lower():
                return parts[9]  # inode 반환
    return None

def find_process_by_inode(inode):
    """주어진 inode를 사용하는 프로세스를 찾습니다."""
    for pid in filter(str.isdigit, os.listdir('/proc')):
        fd_dir = f'/proc/{pid}/fd'
        try:
            for fd in os.listdir(fd_dir):
                link = os.readlink(os.path.join(fd_dir, fd))
                if f'socket:[{inode}]' in link:
                    return pid, get_process_name(pid)
        except (FileNotFoundError, PermissionError):
            continue
    return None, None

def get_process_name(pid):
    """프로세스의 이름을 가져옵니다."""
    try:
        with open(f'/proc/{pid}/comm', 'r') as comm_file:
            return comm_file.read().strip()
    except FileNotFoundError:
        return None

def find_process_using_port(port):
    """특정 포트를 사용하는 프로세스를 찾습니다."""
    hex_port = convert_port_to_hex(port)
    files = ['/proc/net/tcp', '/proc/net/udp', '/proc/net/tcp6', '/proc/net/udp6']
   
    for file in files:
        try:
            inode = read_network_file(file, hex_port)
            if inode:
                pid, comm = find_process_by_inode(inode)
                if pid:
                    return pid, comm, file.split('/')[-1]
            else:
                print(f"{file}에는 해당하는 프로세스 ID가 없습니다.")
        except Exception as e:
            print(e)
    return None, None, None

# 사용 예시
pid, comm, protocol = find_process_using_port(port)

if pid:
    print(f"포트 {port}는 프로세스 {pid} ({comm})에 의해 사용 중입니다. 프로토콜: {protocol}")
else:
    print(f"포트 {port}를 사용 중인 프로세스를 찾을 수 없습니다.")


while True:
    pid, comm, protocol = find_process_using_port(port)
    if not pid: break
    result = subprocess.run(["kill","-9", pid], capture_output=True, text=True)
    print(result)