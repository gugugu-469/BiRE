
import json
import re
import os


# ##  p2so模型


import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('./codes')
import os
import shutil
import argparse
import torch

import torch.utils.data as Data
from torch import nn
from d2l import torch as d2l
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModelForMaskedLM
MODEL_CLASS = {
    'bert': (BertTokenizerFast, BertModel),
    'roberta': (AutoTokenizer, AutoModelForMaskedLM),
    'mcbert': (AutoTokenizer, AutoModelForMaskedLM),
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='./ACE05-DyGIE/processed_data', type=str,
                        help="The task data directory.")
    # 方法名：baseline required=True
    parser.add_argument("--read_dir", default='', type=str,required=True,
                        help="The name of method.")

    parser.add_argument("--data_type", default='test', type=str,
                        help="The name of method.")
                        
    parser.add_argument("--with_type", action="store_true",
                        help="The name of method.")
    # py版本
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    
    gold_path = os.path.join(args.data_dir,args.data_type+'.json')
    predict_path1 = args.read_dir


    # 黄金数据
    all_gold_jsons=[]
    with open(gold_path, 'r') as f_1:
        lines = f_1.readlines()
        for line in lines:
            all_gold_jsons.append(json.loads(line))
    gold_spos=[]
    for i in range(len(all_gold_jsons)):
        gold_json=all_gold_jsons[i]
        spo_list=gold_json['entity_list']
        for spo in spo_list:
            if args.with_type:
                gold_spos.append((i,spo['entity_type'],spo['entity'].strip()))
            else:
                gold_spos.append((i,spo['entity'].strip()))

    #获取预测数据
    all_predict_jsons=[]
    with open(predict_path1, 'r') as f_2:
        lines = f_2.readlines()
        for line in lines:
            all_predict_jsons.append(json.loads(line))
    predict_spos=[]
    for i in range(len(all_predict_jsons)):
        predict_json=all_predict_jsons[i]
        spo_list=predict_json['entity_list']
        for spo in spo_list:
            if args.with_type:
                predict_spos.append((i,spo['entity_type'],spo['entity'].strip()))
            else:
                predict_spos.append((i,spo['entity'].strip()))

    # 计算pre,rec,f1
    P = len(set(predict_spos) & set(gold_spos)) / len(set(predict_spos))
    R = len(set(predict_spos) & set(gold_spos)) / len(set(gold_spos))
    if P+R ==0 :
        F = 0
    else:
        F = (2 * P * R) / (P + R)
    print('\n分数计算完毕：\n')
    print(str(round(P*100,2))+'|'+str(round(R*100,2))+'|'+str(round(F*100,2))+'|'+'\n')


if __name__ == '__main__':
    main()
