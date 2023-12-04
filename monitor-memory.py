import sys
import psutil
from pathlib import Path
import time
import re


def get_memory_usage(pid):
    try:
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        return memory_info.rss  # Resident Set Size (RSS) in bytes
    except psutil.NoSuchProcess as e:
        print(f"Exiting as process {pid} no longer exists")
        sys.exit(0)


def write_memory_usage_to_file(pids, filename, interval=0.2):
    with open(filename, 'w') as file:
        file.write("Time (s),Memory Usage (bytes)\n")

        start_time = time.time()
        while True:
            memory_usage = sum([get_memory_usage(pid) for pid in pids])
            if memory_usage is not None:
                elapsed_time = time.time() - start_time
                file.write(f"{elapsed_time},{memory_usage}\n")
            time.sleep(interval)


if __name__ == "__main__":
    pid_files = Path(".").glob("process-*.txt")
    pids = [int(re.findall(r'\d+', pid_file.name)[0])
            for pid_file in pid_files]
    if len(pids) == 0:
        print("Warning: Did not find and process-*.txt files")

    # Replace 'output_memory_usage.csv' with the desired output file name
    output_file = 'memory_usage.csv'

    # Monitor the process and write memory usage to the file every second for 10 seconds
    write_memory_usage_to_file(pids, output_file, interval=0.2)
