import json
from collections import defaultdict

# 读取JSON数据
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 累加相同name的dur
def aggregate_durations(data):
    durations = defaultdict(int)
    for item in data:
        name = item['name']
        dur = item['dur']
        durations[name] += dur
    return durations

# 累加相同args["op_name"]的dur
def aggregate_durations_opname(data):
    durations = defaultdict(int)
    for item in data:
        op_name = item['args']['op_name']
        dur = item['dur']
        durations[op_name] += dur
    return durations

# 排序并输出为JSON
def sort_and_write_json(durations, output_file):
    sorted_durations = sorted(durations.items(), key=lambda x: x[1], reverse=True)
    with open(output_file, 'w') as file:
        json.dump(sorted_durations, file, indent=4)

# 主函数
def main(input_file, output_file):
    # 读取数据
    data = read_json_file(input_file)

    # 累加dur
    durations = aggregate_durations(data)
    durations = aggregate_durations_opname(data)

    # 排序并写入新文件
    sort_and_write_json(durations, output_file)
    print(f"Data has been processed and saved to {output_file}")

if __name__ == "__main__":
    input_file = 'onnxruntime_profile_10-11.json'  # 输入文件名
    output_file = 'output1011opname.json'  # 输出文件名
    main(input_file, output_file)