from openai import OpenAI
import yaml
import time

# 使用 OpenAI SDK 调用 ECNU 大语言模型平台的函数
def call_openai_sdk(api_key, base_url, model, messages):
    # 创建 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,  # API 密钥
        base_url=base_url,  # 基础 URL
    )
    # 创建聊天完成请求
    completion = client.chat.completions.create(
        model=model,  # 模型名称
        messages=messages,  # 消息列表
    )
    # 返回 JSON 格式的响应
    return completion.choices[0].message.content

config_path = "/root/contract2solidity/llm_api_config.yaml"


'''
   调用 ChatECNU 大模型
    1. 从配置文件中读取 API 配置信息
    2. 依次尝试调用每个 API，记录调用时间和结果
    3. 返回第一个成功调用的结果
    4. 如果所有 API 调用均失败，抛出异常提示用户检查配置和网络连接
'''
def call(messages):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        return f"错误：找不到配置文件 {config_path}"
    
    # print(config)
    
    for api_model in config.keys():
        # print(api_model)
        api_key = config[api_model].get('api_key')
        base_url = config[api_model].get('base_url')
        model_name = config[api_model].get('model_name')
        try:
            start_time = time.time()  # 记录开始时间
            response = call_openai_sdk(api_key, base_url, model_name, messages)
            end_time = time.time()  # 记录成功时的结束时间
            elapsed_time = end_time - start_time
            print(f"✅ 调用 {api_model} 成功！耗时: {elapsed_time:.2f} 秒")
            return response
        except Exception as e:
            print(f"调用 {api_model} 失败: {e}")
    
    raise Exception("Too Many Requests for your account")

def embedding(input: str):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        return f"错误：找不到配置文件 {config_path}"
    
    for api_model in config.keys():
        api_key = config[api_model].get('api_key')
        base_url = config[api_model].get('base_url')
        try:
            client = OpenAI(
                        base_url=base_url,
                        api_key=api_key
                    )
            response = client.embeddings.create(
                        model="ecnu-embedding-small",  # 指定模型
                        input=input,           # 
                    )
            return response.data[0].embedding
        
        except Exception as e:
            print(f"调用 {api_model} 失败: {e}")
    
    raise Exception("Too Many Requests for your account")

def call_example():
    api_key = 'sk-07058a8bd7af45d89b259c9630af531f'  # API 密钥
    base_url = "https://chat.ecnu.edu.cn/open/api/v1"  # 基础 URL
    model = "ecnu-max"  # 模型名称
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},  # 系统消息
        {'role': 'user', 'content': 'Introduce yourself'}  # 用户消息
    ]
    print(messages)
    # 调用使用 OpenAI SDK 的函数并打印结果
    print(call_openai_sdk(api_key, base_url, model, messages))

def embedding_example():
    vec = embedding("Hello, world!")
    print(vec.__len__())
    print(vec[:5])  # 打印前5维度的向量值

# 示例调用
if __name__ == "__main__":
    # for _ in range(10):
    #     print(call([
    #         {'role': 'system', 'content': 'You are a helpful assistant.'},
    #         {'role': 'user', 'content': 'Introduce yourself'}
    #     ]))
    # call_example()
    embedding_example()