# coding=utf-8
"""
多线程并发调用openai api 发压LLM进行数据标注
"""
import openai
import concurrent.futures
import threading
import queue
import time
import json
import sys

##############################  CONFIG  ###################################
openai.api_key = ""
openai.api_base = ""

temperature = 0.01   # 生成结果确定性
top_p = 1   # 对生成回复的概率进行截断
max_tokens = 150   # 生成回复的最大长度

model = "gpt-4"

# 发压配置
NUM_WORKER = 4   # 并发worker线程数量
MAX_QPS = 1    # 每秒发送多少个请求到api
TIME_INTERVAL = 1 / MAX_QPS + 0.1    # 每个请求之间的最小时间间隔
MAX_SAVE_STEP = 100    # 每处理完多少条数据打印一次进度
###########################################################################

prompt_template = '你是一个经验丰富的标注人员，请你对输入的内容进行标注，判断url能否满足对应query（0/1）。\
url为query的满足结果定义为，xxx。\
与query和url共同输入的还有url的content片段、xxx等等，请合理利用这类信息。请以json格式输出结果。\
示例：输入：{{“query“:“xxx“, “url“:“xxx“, “内容片段“:“xxx“}}。\
输出：{{“推理过程”:“xxx”, “标注结果”:“1”}}。\
请严格遵循示例中的输出格式，不要输出其他内容。现在请对如下内容进行标注:{}'


def get_response(messages, temperature, top_p, max_tokens, model):
    response = openai.ChatCompletion.create(model=model,
        messages=messages,
        temperature=temperature, 
        top_p=top_p,
        max_tokens=max_tokens,
    )
    if not response.get("error"):   # 检查回复中是否有错误信息，如果没有错误则继续下面的操作
        return response["choices"][0]["message"]["content"]   # 从回复字典中获取第一个回复的内容并返回
    return response["error"]["message"]   # 如果有错误信息，则返回错误信息


finish_cnt = 0
def worker(input_queue, lock, last_request_time, output_queue):
    """
    单个进程发压函数。

    Args:
        input_queue (Queue): 输入队列，用于接收输入数据。
        lock (threading.Lock): 线程锁，用于保证线程安全。
        last_request_time (list): 列表，用于记录上一次请求的时间。
        output_queue (Queue): 输出队列，用于存储输出数据。

    Returns:
        None
    """
    global finish_cnt
    while not input_queue.empty():
        try:
            line_dic = input_queue.get()
            final_prompt = prompt_template.format(line_dic)
            messages = [
                {"role": "user", "content": final_prompt}
            ]
            with lock:
                current_time = time.time()
                elapsed_time = current_time - last_request_time[0]
                if elapsed_time < TIME_INTERVAL:
                    time.sleep(TIME_INTERVAL - elapsed_time)
                last_request_time[0] = time.time()
            response = get_response(messages, temperature, top_p, max_tokens, model)
            response = response.replace("\n", "")
            output_queue.put((line_dic, response))
        except Exception as e:
            output_queue.put((line_dic, "Error: " + str(e)))
        finally:
            with lock:
                finish_cnt += 1
                if finish_cnt % MAX_SAVE_STEP == 0:
                    print("finish: ", finish_cnt)


def write_output(output_queue, lock):
    """
    将输出队列中的数据写入到文件中，并在队列大小超过100条时触发写入操作。

    Args:
        output_queue (Queue): 包含待写入文件的输出数据的队列，每个元素为一个包含两个元素的元组，第一个元素为字符串类型的行数据，第二个元素为响应数据（可以是字符串或其他可转换为字符串的类型）。
        lock (Lock): 线程锁，用于在写入文件时确保线程安全。

    Returns:
        该函数没有返回值，直接操作文件写入。
    """
    while True:
        if output_queue.qsize() >= MAX_SAVE_STEP:  # 超过 MAX_SAVE_STEP 条保存一下
            with lock:
                with open(output_file, "a") as f:
                    while not output_queue.empty():
                        line_dic, response = output_queue.get()
                        json_line = json.dumps(line_dic, ensure_ascii=False)
                        f.write(json_line + "\t" + response + "\n")
                    f.flush()


def main(input_file, output_file):
    """
    主函数，用于处理输入文件并生成输出文件。

    Args:
        input_file (str): 输入文件的路径。
        output_file (str): 输出文件的路径。

    Returns:
        None: 此函数不返回任何值，但会生成并写入到指定的输出文件中。
    """
    lock = threading.Lock()
    last_request_time = [time.time()]
    output_queue = queue.Queue()
    input_queue = queue.Queue()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKER) as executor:
        threading.Thread(target=write_output, args=(output_queue, lock)).start()
        with open(input_file, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:   # 跳过header
                columns = line.strip().split('\t')   # 每列用\t分隔
                query = columns[0]
                url = columns[1]
                para = columns[2]
                content_fragment = para[:512]   # 截取内容片段

                json_obj = {
                    'query': query,
                    'url': url,
                    '内容片段': content_fragment
                }

                input_queue.put(json_obj)
                
            for _ in range(NUM_WORKER):
                executor.submit(worker, input_queue, lock, last_request_time, output_queue)
        executor.shutdown(wait=True)

    # 刷下输出区
    with lock:
        with open(output_file, "a", encoding='utf-8') as f:
            while not output_queue.empty():
                line_dic, response = output_queue.get()
                json_line = json.dumps(line_dic, ensure_ascii=False)
                f.write(json_line + "\t" + response + "\n")
            f.flush()


if __name__ == '__main__':
    '''
    命令行输入 python press_llm.py input_data infer_res >> press_log
    每infer 100条，press_log 会打印一行 finish
    '''
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    main(input_file, output_file)