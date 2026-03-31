from __future__ import absolute_import, division, print_function
import sys
sys.path.append('..')
import argparse
import json
import os
import time
import warnings
from datetime import datetime
import tiktoken
import torch
import requests
import pickle
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.logger import MyLogger
from utils.retrieve import init_bert_model, query
from llm_call import call
from utils.custom_tools import fix_missing_brackets

# 保存测试prompt
prompt_idx = 0
prompt_list = []

def few_shot_generation(args, prompt, tokenizer, model, sample):
    output_list = []
    
    if args.model == "ecnu-max" or args.model.startswith("test"):
        messages = [
                    {"role": "system",
                     "content": "You are a professional Solidity engineer. Please continue to generate a function based on the provided requirement and function signature, NO need to repeat the signature. End your function with // END_OF_FUNCTION. Never add any additional explanation or comments."},
                    {"role": "user", "content": prompt},
                ]
        # logger.info_blue("--------------prompt(start)-----------------")
        # logger.info(prompt)
        # logger.info_blue("--------------prompt(end)-----------------")
        
        for idx in range(sample):
            response = call(messages)
            output = str(response)
            output = output[:output.rfind("// END_OF_FUNCTION")]
            output = output.replace("```solidity", "")
            output = function_full_sig.strip('\n') + '\n' + output.strip('\n')
            
            logger.info_blue("--------------function(start)-----------------")
            # 修复缺失的右括号
            output = fix_missing_brackets(output)
            logger.info_blue(output)
            logger.info_blue("--------------function(end)-----------------")
            with open(
                    f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx}.txt",
                    'w') as f:
                f.write(output)
            output_list.append(output)    
            
    elif args.model == "debug":
        return []
    
    return output_list


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# /root/contract2solidity/SolEval/data/example.json

if __name__ == '__main__':
    with open('/root/contract2solidity/SolEval/data/dataset.json', 'r') as file:
        data = json.load(file)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    total_inference_time = 0
    inference_tries = 0
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="debug")
    parser.add_argument('--k',
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering. '
                             'Only applies for sampling mode, with range from 1 to 100.',
                        type=int, default=50)
    parser.add_argument('--p',
                        help='Only the most probable tokens with probabilities that add up to top_p or higher are considered during decoding. '
                             'The valid range is 0.0 to 1.0. 1.0 is equivalent to disabled and is the default. '
                             'Only applies to sampling mode. Also known as nucleus sampling.',
                        type=float, default=0.95)
    parser.add_argument('--temperature',
                        help='What sampling temperature to use, between 0 and 2. '
                             'Higher values like 0.8 will make the output more random, '
                             'while lower values like 0.2 will make it more focused and deterministic.',
                        type=float, default=1)
    parser.add_argument('--context', action='store_true', default=False, help='Enable context for generation')
    parser.add_argument('--testcase', action='store_true', default=False, help='Enable testcase for generation')
    parser.add_argument('--shot', help='', type=int, default=2)
    parser.add_argument('--filesize', help='测试文件数量', type=int, default=81)
    parser.add_argument('--methodsize', help='测试方法数量', type=int, default=1125)
    parser.add_argument("--overwrite", action='store_true', default=False, help='Whether to overwrite existing patch files')
    
    args = parser.parse_args()
    embedding_list, original_document_list, func_list = init_bert_model()
    log_file = f"log_{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}_{current_time}.txt"
    logger = MyLogger(f"logs_patch/rag/{log_file}")
    total_token = 0
    token_tries = 0
    file_size = args.filesize
    method_size = args.methodsize
    
    tokenizer, model = None, None

    if not os.path.exists(f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}"):
        os.makedirs(f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}")



    real_path_cargo = pickle.load(open("/root/contract2solidity/SolEval/prebuilt/real_path_cargo.pkl", "rb"))
    
    context_example_cnt = 0  
    
    if args.model == "debug":
        # 1. 基础数量统计
        total_files = len(data)
        # 统计每个文件包含的方法数量
        methods_per_file = [len(content) for content in data.values()]
        total_methods = sum(methods_per_file)
        
        logger.info_blue(f"--- 数据集概览 ---")
        logger.info_blue(f"总文件数: {total_files}")
        logger.info_blue(f"总方法数: {total_methods}")
        if total_files > 0:
            logger.info_blue(f"平均每个文件的方法数: {total_methods / total_files:.2f}")
            logger.info_blue(f"单文件最多方法数: {max(methods_per_file)}")
        
        # 构建映射关系
        pattern_to_path = {}
        for file_path, file_content in tqdm(data.items(), colour='green'):
            for method in tqdm(file_content, colour="red"):
                identifier = method['identifier']
                pattern = f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_*"
                
                
        sys.exit(0)
    
    file_count = 0
    method_count = 0
    overwrite = args.overwrite
    
    for file_path, file_content in tqdm(data.items(), colour='green'):
        logger.info_white("file_path:\n" + file_path)
        
        # 限制测试文件数量
        file_count += 1
        if file_count > file_size or method_count > method_size:
            break
        
        for method in tqdm(file_content, colour="red"):
            
            method_count += 1
            # 限制测试方法数量
            if method_count > method_size:
                break
            
            identifier = method['identifier']
            
            # 查看是否覆写
            file_name = f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{args.sample - 1}.txt"
            if os.path.exists(file_name):
                if overwrite:
                    # 删除该文件
                    os.remove(file_name)
                else:
                    logger.info_blue(
                        f"exist patch file, skipping {real_path_cargo[file_path].split('/')[-1]}_function_{identifier}")
                    continue

            comment = method['human_labeled_comment'].strip()
            context = eval(method['context']) if method['context'] != "" else None
            # if context == set():
            # context = None
            function_full_sig = method['full_signature'].strip() + ' {' + '\n'

            prompt = ""
            examples = query(comment, embedding_list, original_document_list, func_list, args.shot)
            for example in examples:
                prompt = prompt + "// IMPLEMENT THE FUNCTIONALITY BASED ON THE PROVIDED REQUIREMENT.\n\n// START_OF_REQUIREMENT\n" + example[0] + "\n// END_OF_REQUIREMENT\n\n" + "// START_OF_CONTEXT" + '\n' + example[2] + "\n// END_OF_CONTEXT" + '\n\n'+"// START_OF_FUNCTION\n" + example[1] + "\n// END_OF_FUNCTION\n\n"
            prompt = prompt + "// IMPLEMENT THE FUNCTIONALITY BASED ON THE PROVIDED REQUIREMENT.\n\n// START_OF_REQUIREMENT\n" + comment + "\n// END_OF_REQUIREMENT\n"

            if args.context:
                prompt = prompt + '\n' + "// START_OF_CONTEXT" + '\n'
                if not context:
                    prompt = prompt + "No context for this function" + '\n'
                else:
                    context_example_cnt += 1
                    for c in context:
                        prompt = prompt + c + '\n'
                prompt = prompt + "// END_OF_CONTEXT" + '\n'
                
                
            prompt = prompt + '\n' + "// START_OF_FUNCTION" + '\n' + function_full_sig
            
            prompt_idx += 1
            prompt_list.append({
                "idx": prompt_idx,
                "prompt": prompt
            })
            
            import glob
            import os
            import re
            have_sample = 0
            pattern = f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_*"
            matching_files = []
            for file_path_ in glob.glob(pattern):
                if os.path.isfile(file_path_) and re.search(rf'_{identifier}_\d+\.txt$', file_path_):
                    matching_files.append(file_path_)
            have_sample = len(matching_files)


            while True:
                try:
                    start_time = time.time()
                    output_list = few_shot_generation(args, prompt, tokenizer, model, args.sample - have_sample)
                    end_time = time.time()
                    inference_tries += args.sample - have_sample
                    total_inference_time += end_time - start_time
                    average_inference_time = total_inference_time / inference_tries
                    logger.info_green("average_inference_time: {:.2f}s".format(average_inference_time))
                    break
                except Exception as e:
                    print(e)
                    if "Too Many Requests for your account" in str(e):
                        for _ in tqdm(range(120), desc="sleeping", colour="yellow"):
                            time.sleep(1)
                        # slow = True
            output_list = [function_full_sig.strip('\n') + '\n' + output.strip('\n') for output in output_list]
            
            for idx, out in enumerate(output_list):
                with open(
                        f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx + have_sample}.txt",
                        'w') as f:
                    f.write(out)
            
    
    if args.model == "debug":
        with open('/root/contract2solidity/SolEval/dataset/prompt.json', 'w', encoding='utf-8') as f:
            json.dump(prompt_list, f, ensure_ascii=False, indent=4) 
            
    logger.info_blue(f"Total files processed")