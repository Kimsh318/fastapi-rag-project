import re
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt

# 1. 로그 파일에서 데이터 추출
def extract_log_data(log_file):
    pattern = r"(?P<timestamp>[0-9-]+ [0-9:.]+) - .+ - INFO - (?P<function_name>\w+) completed in (?P<elapsed_time>[0-9.]+) seconds"
    pattern = r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .+ - INFO - (?P<function_name>\w+) completed in (?P<elapsed_time>\d+\.\d+) seconds"

    data = []

    with open(log_file, "r") as file:
        for line in file:
            #print(line)
            match = re.search(pattern, line)
            #print(f"({match} : {line})")
            if match:
                timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f")
                function_name = match.group("function_name")
                elapsed_time = float(match.group("elapsed_time"))
                data.append((timestamp, function_name, elapsed_time))

    return data

# 2. 데이터를 1분 간격으로 그룹화
def group_data_by_minute(data):
    grouped_data = defaultdict(lambda: defaultdict(list))
    start_time = min(entry[0] for entry in data)
    end_time = max(entry[0] for entry in data)
    current_time = start_time

    while current_time <= end_time:
        next_minute = current_time + timedelta(minutes=1)
        for timestamp, function_name, elapsed_time in data:
            if current_time <= timestamp < next_minute:
                grouped_data[current_time][function_name].append(elapsed_time)
        current_time = next_minute

    return grouped_data

# 3. 각 그룹의 평균값 계산
def calculate_average_per_minute(grouped_data):
    average_data = defaultdict(dict)
    for minute, functions in grouped_data.items():
        for function_name, times in functions.items():
            average_data[minute][function_name] = sum(times) / len(times) if times else 0
    return average_data

# 4. 그래프 시각화
def plot_average_data(average_data, output_file="average_elapsed_time.png"):
    plt.figure(figsize=(12, 6))
    functions = set(fn for minute_data in average_data.values() for fn in minute_data.keys())

    for function_name in functions:
        times = []
        averages = []
        for minute, functions in average_data.items():
            times.append(minute)
            averages.append(functions.get(function_name, 0))

        plt.plot(times, averages, label=function_name)

    plt.xlabel("Time (minute)")
    plt.ylabel("Average elapsed time (seconds)")
    plt.title("Average Elapsed Time per Minute")
    plt.legend()
    plt.grid()

    # 그래프를 파일로 저장
    plt.savefig(output_file)
    print(f"Graph saved as {output_file}")
    plt.close()

# 5. 실행
if __name__ == "__main__":
    log_file = "./sample_app.log"  # 로그 파일 경로
    log_file= "/app/tmp_test/logging_test/sample_app.log"
    log_data = extract_log_data(log_file)
    print(log_data)
    grouped_data = group_data_by_minute(log_data)
    average_data = calculate_average_per_minute(grouped_data)
    plot_average_data(average_data)
