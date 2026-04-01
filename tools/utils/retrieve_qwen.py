import sys
from pprint import pprint

sys.path.append('..')
import copy
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from llm_call import embedding  # 确保该函数返回 1024 维的 list 或 ndarray

# 虽然不用本地模型，但保留 device 用于加速余弦相似度计算（Top-K）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_qwen_embedding():
    """
    初始化并将所有文档向量化。使用外部 embedding 函数替代本地推理。
    """
    print("Initializing embeddings via llm_call.embedding...")
    document_list = []
    func_list = []
    
    # 1. 加载数据
    with open('/root/contract2solidity/SolEval/data/example.json', 'r') as file:
        data = json.load(file)
        
    for file_path, file_content in tqdm(data.items(), desc="Parsing JSON"):
        for method in file_content:
            comment = method['human_labeled_comment'].strip()
            # 统一注释结尾格式
            if not comment.endswith("\n */"):
                comment += "\n */"
            document_list.append(comment)
            func_list.append(method)
            
    original_document_list = copy.deepcopy(document_list)
    
    # 2. 检查缓存（建议缓存文件名体现出 1024 维特性）
    cache_file = '../prebuilt/qwen_1024_embeddings.npy'
    
    if os.path.exists(cache_file):
        print("Loading embeddings from cache...")
        cls_embeddings = np.load(cache_file)
    else:
        print("Generating embeddings (this may take a while depending on API speed)...")
        all_vecs = []
        
        # 逐条调用 embedding 函数
        for text in tqdm(document_list, desc="Calling Embedding API"):
            # 假设 embedding(text) 返回类似 [0.1, 0.2, ..., 1024维度]
            vec = embedding(text)
            all_vecs.append(vec)
        
        cls_embeddings = np.array(all_vecs, dtype=np.float32)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, cls_embeddings)
    
    # 转为 tensor 方便后续在 GPU 上做大规模相似度计算
    cls_embeddings_tensor = torch.tensor(cls_embeddings).to(device)
    
    return cls_embeddings_tensor, original_document_list, func_list


def query(input_requirements: str, embedding_list: torch.Tensor, original_document_list: list, func_list: list, k: int) -> list:
    """
    查询接口。调用 embedding 函数获取 query 向量并计算相似度。
    """
    # 保持 Qwen 建议的 Prompt 指令
    query_prompt = "Represent this English code comment for retrieving relevant Solidity implementations: "
    query_text = query_prompt + input_requirements
    
    # 获取输入要求的向量表示 (1024维)
    query_vec_raw = embedding(query_text)
    query_vec = torch.tensor(query_vec_raw, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 计算余弦相似度 (Batch 计算)
    # embedding_list: (N, 1024), query_vec: (1, 1024)
    cosine_similarity = torch.nn.functional.cosine_similarity(query_vec, embedding_list, dim=1)
    
    # 获取 TopK
    _, topk_indices = torch.topk(cosine_similarity, k)
    
    result = []
    for index in topk_indices:
        idx = index.item()
        original_doc = original_document_list[idx]
        func_dict = func_list[idx]
        
        context = func_dict.get("context", "No context for this function")
        if context == "set()": 
            context = "No context for this function"
        
        result.append([
            original_doc, 
            func_dict.get("body", ""),
            context
        ])
        
    return result

if __name__ == '__main__':
    # 修正了调用名，统一为 init_qwen_embedding
    embedding_list, original_document_list, func_list = init_qwen_embedding()
    
    inpu_test = "/**\n * @notice Packs a `bytes2` and a `bytes10` into a single `bytes12` value.\n *\n * @dev This function uses inline assembly to perform bitwise operations to combine the two input bytes.\n * - The `bytes2` value is shifted left by 240 bits and masked to ensure it occupies the correct position.\n * - The `bytes10` value is shifted left by 176 bits and masked to ensure it occupies the correct position.\n * - The two values are then combined using a bitwise OR operation.\n *\n * @param left The `bytes2` value to be packed into the higher-order bits of the result.\n * @param right The `bytes10` value to be packed into the lower-order bits of the result.\n * @return result The combined `bytes12` value containing both `left` and `right`.\n *\n * Steps:\n * 1. Mask and shift the `left` value to align it with the higher-order bits of the result.\n * 2. Mask and shift the `right` value to align it with the lower-order bits of the result.\n * 3. Combine the two values using a bitwise OR operation to produce the final `bytes12` result.\n"
    
    k = 2
    pprint(query(inpu_test, embedding_list, original_document_list, func_list, k))