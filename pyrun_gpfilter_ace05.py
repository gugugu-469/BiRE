
# ##  er模型


import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('./codes')
import os
import argparse
import torch
import numpy as np
torch.set_printoptions(threshold=np.inf)
import shutil

from models import GPFilterModel
from trainer import GPFilterTrainer
from data import GPFilterDataset, GPFilterDataProcessor
from utils import init_logger, seed_everything, get_devices, get_time

import torch.utils.data as Data
from torch import nn
from d2l import torch as d2l
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModelForMaskedLM, RobertaModel, AlbertModel, AlbertTokenizerFast
MODEL_CLASS = {
    'bert': (BertTokenizerFast, BertModel),
    'roberta': (AutoTokenizer, RobertaModel),
    'mcbert': (AutoTokenizer, AutoModelForMaskedLM),
    'albert':(AlbertTokenizerFast, AlbertModel)
}


def get_args():
    parser = argparse.ArgumentParser()
    
    # 方法名：
    parser.add_argument("--method_name", default='', type=str,
                        help="The name of method.")
    
    # 数据集存放位置：
    parser.add_argument("--data_dir", default='./', type=str,
                        help="The task data directory.")
    
    # 增强数据集存放位置：
    parser.add_argument("--enhanced_data_dir", default='./enhanced_data', type=str,
                        help="The path of enhanced_data produced by doing dual eval with test set.")
    
    # 是否做数据增强
    parser.add_argument("-do_enhance", default=True, type=bool,
                    help="Whether to do data enhance.")
    
    # 预训练模型存放位置:
    parser.add_argument("--model_dir", default='/root/nas/Models', type=str,
                        help="The directory of pretrained models")
    
    # 模型类型:
    parser.add_argument("--model_type", default='bert', type=str, 
                        help="The type of selected pretrained models.")
    
    # 预训练模型:
    parser.add_argument("--pretrained_model_name", default='RoBERTa_zh_Large_PyTorch', type=str,
                        help="The path or name of selected pretrained models.")
    
    # 微调模型:
    parser.add_argument("--finetuned_model_name", default='gpfilter', type=str,
                        help="The name of finetuned model")
    
    # 微调模型参数存放位置：
    parser.add_argument("--output_dir", default='./checkpoint', type=str,
                        help="The path of result data and models to be saved.")
    
    # 是否训练：True
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    
    # 是否预测：False required=True
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run the models in inference mode on the test set.")
    
    # 是否做过滤器：False required=True
    parser.add_argument("--do_filter", action="store_true",
                        help="Whether to run the models to filter results from different methods.")
    
    # 预测时加载的模型版本，如果做预测，该参数是必需的
    parser.add_argument("--model_version", default='', type=str,
                        help="model's version when do predict")
    
    # 提交结果保存目录：./result_output required=True
    parser.add_argument("--result_output_dir", default='./result_output', type=str,
                        help="the directory of commit result to be saved")
    
    # 设备：-1：CPU， i：cuda:i(i>0), i可以取多个，以逗号分隔 required=True
    parser.add_argument("--devices", default='1', type=str,
                        help="the directory of commit result to be saved")
    
    parser.add_argument("--loss_show_rate", default=200, type=int,
                        help="liminate loss to [0,1] where show on the train graph")
    
    # 序列最大长度：
    parser.add_argument("--max_length", default=256, type=int,
                        help="the max length of sentence.")
    
    # 训练batch_size：
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    
    # 评估batch_size：
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size for evaluation.")
    
    # 学习率：
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    
    # 权重衰退：
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    
    # 极小值：
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    
    # epochs
    parser.add_argument("--epochs", default=40, type=int,
                        help="Total number of training epochs to perform.")
    
    # 线性学习率比例：
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for, "
                             "E.g., 0.1 = 10% of training.")
    
    # earlystop_patience：
    parser.add_argument("--earlystop_patience", default=100, type=int,
                        help="The patience of early stop")
    
    # 多少step后打印一次：
    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    
    
    # 随机数种子：
    parser.add_argument('--seed', type=int, default=2021,
                        help="random seed for initialization")
    
    # 训练时保存 save_metric 最大存取模型 
    parser.add_argument("--save_metric", default='f1', type=str,
                        help="the metric determine which model to save.")
    
    # 是否做rdrop（变相的数据增强）
    parser.add_argument('--do_rdrop', action="store_true",
                        help="whether to do r-drop")

    # 是否跳过eval
    parser.add_argument('--no_eval', action="store_true",
                        help="whether to do eval")
    
    # rdrop 中的参数，alpha越大则loss越偏向kl散度
    parser.add_argument('--rdrop_alpha', type=int, default=4,
                        help="hyper-parameter in rdrop")
    
    # 正则化手段，dropout
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="dropout rate")
    
    # 正则化手段，filter_threshold
    parser.add_argument('--filter_threshold', type=float, default=0,
                        help="the threshold to filer results from different methods")
    
    # gplinker中的隐藏层维度
    parser.add_argument('--inner_dim', type=int, default=64,
                        help="inner dim of gplinker")
    
    # filter模型的预测阈值：
    parser.add_argument('--predict_threshold', type=float, default=10,
                        help="fliter model's predict_threshold")
    
    # gpner模型版本
    parser.add_argument('--prev_model_1', type=str, default='',
                        help="model version of gpner")
    
    # gpner9模型版本
    parser.add_argument('--prev_model_2', type=str, default='',
                        help="model version of gpner9")

    # head过滤阈值
    parser.add_argument('--filter_head_threshold', type=float, default=-9.5,
                        help="whether to use entity type information")

    # tail过滤阈值
    parser.add_argument('--filter_tail_threshold', type=float, default=-9.5,
                        help="whether to use entity type information")

    # 梯度裁剪
    parser.add_argument('--max_grad_norm', type=float, default=1,
                        help="the rate of gradient trimming")
    # py版本
    args = parser.parse_args()
    args.devices = get_devices(args.devices.split(','))
    args.device = args.devices[0]
    args.distributed = True if len(args.devices) > 1  else False 
    seed_everything(args.seed)
    args.time = get_time(fmt='%m-%d-%H-%M')
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.method_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.pretrained_model_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.finetuned_model_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    print(args.output_dir)    
    args.result_output_dir = os.path.join(args.result_output_dir, args.method_name) 
    if not os.path.exists(args.result_output_dir):
        os.mkdir(args.result_output_dir)
    if args.do_train and args.do_filter:
        args.model_version = args.time
    if args.do_filter == True and args.model_version == '':
        raise Exception('做过滤的话必须提供加载的模型版本')    
    if args.do_filter == True and args.prev_model_1 == '' and args.prev_model_2 == '':
        print('做过滤不提供前两个模型的版本，可能无法找到数据加载文件')
    return args


def main():
    args = get_args()
    logger = init_logger(os.path.join(args.output_dir, 'log.txt'))
    tokenizer_class, model_class = MODEL_CLASS[args.model_type]
    if args.do_train:
        makedirs = os.path.join(args.output_dir, args.model_version)
        if not os.path.exists(makedirs):
            os.makedirs(makedirs)
        logger.info(f'Training {args.finetuned_model_name} model...')
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.pretrained_model_name), do_lower_case=True)

        data_processor = GPFilterDataProcessor(args)
        train_samples = data_processor.get_train_sample()
        eval_samples = data_processor.get_dev_sample()
        train_dataset =GPFilterDataset(train_samples, data_processor, tokenizer, args, mode='train')
        eval_dataset = GPFilterDataset(eval_samples, data_processor, tokenizer, args, mode='eval')

        model = GPFilterModel(model_class, args)
        trainer = GPFilterTrainer(args=args, model=model, data_processor=data_processor,
                            tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                            logger=logger)

        global_step, best_step = trainer.train()
        
    if args.do_filter:
        load_dir = os.path.join(args.output_dir, args.model_version)
        logger.info(f'load tokenizer from {load_dir}')
        tokenizer = tokenizer_class.from_pretrained(load_dir)

        data_processor = GPFilterDataProcessor(args)

        model = GPFilterModel(model_class, args)
        trainer = GPFilterTrainer(args=args, model=model, data_processor=data_processor,
                            tokenizer=tokenizer, logger=logger)
        trainer.load_checkpoint()
        trainer.predict_filter()


if __name__ == '__main__':
    main()





