import torch
import torch.nn as nn
import os
import json
import jsonlines
import shutil
import math
import numpy as np
import torch.nn.functional as F
from d2l import torch as d2l
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import ProgressBar, TokenRematch, get_time, save_args, SPO, ACESPO
from metrics import er_metric, re_metric, gen_metric, rc_metric, p2so_metric
from loss import multilabel_categorical_crossentropy, sparse_multilabel_categorical_crossentropy
from optimizer import GPLinkerOptimizer
import wandb

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]


class Trainer(object):
    def __init__(
            self,
            args,
            data_processor,
            logger,
            model=None,
            tokenizer=None,
            train_dataset=None,
            eval_dataset=None,
    ):

        self.args = args
        self.model = model
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        

        if train_dataset is not None and isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset

        if eval_dataset is not None and isinstance(eval_dataset, Dataset):
            self.eval_dataset = eval_dataset

        self.logger = logger

    def train(self):
        args = self.args
        logger = self.logger
        model = self.model
        epoch_best_f1 = 0
        self.output_dir = os.path.join(args.output_dir, args.model_version)

        
        if args.distributed == True:
            model = nn.DataParallel(model, device_ids=args.devices).to(args.device)
        else:
            model.to(args.device)
            
        
        
        train_dataloader = self.get_train_dataloader()

        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = num_training_steps * args.warmup_proportion
        num_examples = len(train_dataloader.dataset)
        
        optimizer = GPLinkerOptimizer(args, model, train_steps= len(train_dataloader)  * args.epochs)

        logger.info("***** Running training *****")
        logger.info("Num samples %d", num_examples)
        logger.info("Num epochs %d", args.epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_warmup_steps)

        global_step = 0
        best_step = None
        best_score = -1
        cnt_patience = 0
        
        animator = d2l.Animator(xlabel='epoch', xlim=[0, args.epochs], ylim=[0, 1], fmts=('k-', 'r--', 'y-.', 'm:', 'g--', 'b-.', 'c:'),
                                legend=[f'train loss/{args.loss_show_rate}', 'train_p', 'train_r', 'train_f1', 'val_p', 'val_r', 'val_f1'])
        # 统计指标
        metric = d2l.Accumulator(5)
        num_batches = len(train_dataloader)
        
        
        
        for epoch in range(args.epochs):
            print('Now Epoch:{}'.format(epoch))
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, item in enumerate(train_dataloader):
                loss, train_p, train_r, train_f1 = self.training_step(model, item)
                loss = loss.item()
                metric.add(loss, train_p, train_r, train_f1, 1)
                pbar(step, {'loss': loss})

                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.no_eval:
                        animator.add(
                                global_step / num_batches, 
                                (loss / args.loss_show_rate, train_p, train_r, train_f1, 0, 0, 0))
                        if not os.path.exists(self.output_dir):
                            os.makedirs(self.output_dir)
                        d2l.plt.savefig(os.path.join(self.output_dir, '训练过程.jpg'), dpi=300)
                    else:
                        val_p, val_r, val_f1 = self.evaluate(model)
                        animator.add(
                            global_step / num_batches, 
                            (# metric[0] / metric[-1] / args.loss_show_rate, # loss太大，除以loss_show_rate才能在[0,1]范围内看到
                             loss / args.loss_show_rate,
                             train_p,  # metric[1] / metric[-1],
                             train_r,  # metric[2] / metric[-1],
                             train_f1, # metric[3] / metric[-1],
                             val_p,
                             val_r,
                             val_f1))
                        if not os.path.exists(self.output_dir):
                            os.makedirs(self.output_dir)
                        d2l.plt.savefig(os.path.join(self.output_dir, '训练过程.jpg'), dpi=300)

                        if args.save_metric == 'step':
                            save_metric = global_step
                        elif args.save_metric == 'epoch':
                            save_metric = epoch
                        elif args.save_metric == 'loss':
                            save_metric = math.exp(- loss / 10) # math.exp(- metric[0] / metric[-1] / 10)
                        elif args.save_metric == 'p':
                            save_metric = val_p
                        elif args.save_metric == 'r':
                            save_metric = val_r
                        elif args.save_metric == 'f1':
                            save_metric = val_f1

                        if save_metric > best_score:
                            best_score = save_metric
                            best_step = global_step
                            cnt_patience = 0
                            self.args.loss = loss # metric[0] / metric[-1]
                            self.args.train_p, self.args.train_r, self.args.train_f1 = train_p, train_r, train_f1
                            self.args.val_p, self.args.var_r, self.args.val_f1 = val_p, val_r, val_f1
                            self._save_checkpoint(model)
                        else:
                            cnt_patience += 1
                            self.logger.info("Earlystopper counter: %s out of %s", cnt_patience, args.earlystop_patience)
                            if cnt_patience >= self.args.earlystop_patience:
                                break
            
            if cnt_patience >= args.earlystop_patience:
                break
            if args.no_eval:
                self.args.loss = loss
                self.args.train_p, self.args.train_r, self.args.train_f1 = train_p, train_r, train_f1
                self._save_checkpoint(model)


            # if args.finetuned_model_name in ['gpner9','gpner']:
            #     # 加载一个epoch后的模型并预测计算分数
            #     self.predict()
            #     tmp_F = self.get_epoch_prf()
            #     if tmp_F > epoch_best_f1:
            #         print('FIND BEST F1:{}'.format(tmp_F))
            #         epoch_best_f1 = tmp_F
            #         self._save_best_epoch_checkpoint(model)



        logger.info(f"\n***** {args.finetuned_model_name} model training stop *****" )
        logger.info(f'finished time: {get_time()}')
        logger.info(f"best val_{args.save_metric}: {best_score}, best step: {best_step}\n" )

        return global_step, best_step


    def get_epoch_prf(self):
        data_type = 'dev'
        gold_path = os.path.join(self.args.data_dir,data_type+'.json')
        predict_path1 = os.path.join('./result_output', self.args.finetuned_model_name,self.args.model_version,data_type+'.jsonl')

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
                gold_spos.append((i,spo['entity_type'],spo['entity'].strip()))


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
                predict_spos.append((i,spo['entity_type'],spo['entity'].strip()))
                

        # 计算pre,rec,f1
        if len(set(predict_spos)) == 0:
            P=0
        else:
            P = len(set(predict_spos) & set(gold_spos)) / len(set(predict_spos))
        R = len(set(predict_spos) & set(gold_spos)) / len(set(gold_spos))
        if P+R ==0 :
            F = 0
        else:
            F = (2 * P * R) / (P + R)
        print('\n一个EPOCH分数：\n')
        print(str(round(P*100,2))+'|'+str(round(R*100,2))+'|'+str(round(F*100,2))+'|'+'\n')
        return P,R,F

    def predict(self):
        raise NotImplementedError

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model):
        args = self.args
        
        if args.distributed:
            model=model.module
        # 防止91存到3卡，但是82没有3卡的情况
        model = model.to(torch.device('cpu'))
        torch.save(model.state_dict(), os.path.join(self.output_dir, 'pytorch_model.pt'))
        self.logger.info('Saving models checkpoint to %s', self.output_dir)
        self.tokenizer.save_vocabulary(save_directory=self.output_dir)
        model = model.to(args.device)
        save_args(args, self.output_dir)
        shutil.copyfile(os.path.join(args.model_dir, args.pretrained_model_name, 'config.json'),
                        os.path.join(self.output_dir, 'config.json'))

    def _save_best_epoch_checkpoint(self, model):
        args = self.args
        out = os.path.join(self.output_dir,'best')
        if not os.path.exists(out):
            os.makedirs(out)
        if args.distributed:
            model=model.module
        # 防止91存到3卡，但是82没有3卡的情况
        model = model.to(torch.device('cpu'))
        torch.save(model.state_dict(), os.path.join(out, 'pytorch_model.pt'))
        self.logger.info('Saving models checkpoint to %s', out)
        self.tokenizer.save_vocabulary(save_directory=out)
        model = model.to(args.device)
        save_args(args, out)
        shutil.copyfile(os.path.join(args.model_dir, args.pretrained_model_name, 'config.json'),
                        os.path.join(out, 'config.json'))
    
    
    def load_checkpoint(self):
        args = self.args
        load_dir = os.path.join(args.output_dir, args.model_version)
        self.logger.info(f'load model from {load_dir}')
        # 每次加载到cpu中，防止爆显存
        checkpoint = torch.load(os.path.join(load_dir, 'pytorch_model.pt'), map_location=torch.device('cpu'))
        if 'module' in list(checkpoint.keys())[0].split('.'):
            self.model = nn.DataParallel(self.model, device_ids=args.devices).to(args.device)
        self.model.load_state_dict(checkpoint)
    
    def training_step(self, model, item):
        raise NotImplementedError

    def get_train_dataloader(self):
        collate_fn = self.train_dataset.collate_fn if hasattr(self.train_dataset, 'collate_fn') else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False if self.args.do_rdrop else True,
            collate_fn=collate_fn
        )

    def get_eval_dataloader(self):
        collate_fn = self.eval_dataset.collate_fn if hasattr(self.eval_dataset, 'collate_fn') else None
        return DataLoader(
            self.eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn
        )

    def get_test_dataloader(self, test_dataset, batch_size=None):
        collate_fn = test_dataset.collate_fn if hasattr(test_dataset, 'collate_fn') else None
        if not batch_size:
            batch_size = self.args.eval_batch_size

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )


    
class GPNERTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNERTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_entity_labels = item
        logits = model(batch_token_ids, batch_mask_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        # print('logits shape')
        # print(logits.shape)
        # print('batch_entity_labels')
        # print(batch_entity_labels.shape)
        loss.backward()
        p, r, f1 = self.cal_prf(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        self.predict_for_eval()
        P,R,F = self.get_epoch_prf()
        return P,R,F
    
    def cal_prf(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True):
        args = self.args
        model = self.model
        device = args.device
        num_examples = len(test_samples)
        model.to(device)
        id2class = self.data_processor.id2class
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
            token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=args.max_length, truncation=True)["offset_mapping"]
            new_span, entities = [], []
            for i in token2char_span_mapping:
                if i[0] == i[1]:
                    new_span.append([])
                else:
                    if i[0] + 1 == i[1]:
                        new_span.append([i[0]])
                    else:
                        new_span.append([i[0], i[-1] - 1])
            threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > 0)):
                entities.add((entity_type, h, t))

            entity_list = []
            for entity_type, sh, st in entities:
                entity = text[new_span[sh][0]:new_span[st][-1] + 1]
                temp = {'entity': entity, 'entity_type': id2class[entity_type]}
                entity_list.append(temp)
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self):
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_test_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples)
            for data in predict_data0:
                dic = data
                f.write(dic)

    def predict_entity_for_eval(self):
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_dev_sample_for_eval()
        num_examples = len(test_samples)
        logger.info("***** Running subject EVAL *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'dev_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples)
            for data in predict_data0:
                dic = data
                f.write(dic)
                
    def predict_for_eval(self):
        args = self.args
        logger = self.logger
        self.predict_entity_for_eval()
        sp20_data = self.predict_xp2x(is_sp2o=True if args.finetuned_model_name=='gpnerace05' else False , do_eval=True)
        output_dir = os.path.join(args.result_output_dir, 'dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)

    def predict(self):
        args = self.args
        logger = self.logger
        self.predict_entity()
        sp20_data = self.predict_xp2x(is_sp2o=True if args.finetuned_model_name=='gpnerace05' else False)
        output_dir = os.path.join(args.result_output_dir, 'test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_op2s_sp2o_dual(self):
        args = self.args
        logger = self.logger
        with jsonlines.open(os.path.join('./result_output', 'gpner9', 'test.jsonl'), mode='r') as lines:
            op2s_data = [line for line in lines]
        sp2o_dual_data = self.predict_xp2x(is_sp2o=True, op2s_result=op2s_data)
        output_dir = os.path.join('./result_output', 'gpner9', 'test-dual.jsonl')
        num_examples = 4482
        logger.info("***** Running Dual op2s-sp20 *****")
        logger.info("Num samples %d", num_examples)
        logger.info(f"***** write predict file to {output_dir} *****")
        pbar = ProgressBar(n_total=num_examples, desc='Dual')
        with jsonlines.open(output_dir, mode='w') as f:
            for step, (sp2o, op2s) in enumerate(zip(op2s_data, sp2o_dual_data)):
                pbar(step)
                dic = {'text': sp2o['text'], 'spo_list': []}
                for spo in (set(SPO(spo) for spo in sp2o['spo_list']) & set(SPO(spo) for spo in op2s['spo_list'])):
                    dic['spo_list'].append(spo.spo)
                f.write(dic)

    def predict_xp2x(self, is_sp2o=True, op2s_result=None, do_eval=False):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            if not do_eval:
                with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
                    for line in lines:
                        samples.append(line)
            else:
                with jsonlines.open(os.path.join(args.result_output_dir, 'dev_entity_list.jsonl'), mode='r') as lines:
                    for line in lines:
                        samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if op2s_result == None:
                    for predicate in self.data_processor.predicates:
                        test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                
            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                predicate = data['predicate']
                pre_entity_type = data['entity_type']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                                    'object': post_entity, 'object_type': post_entity_type})
                        
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                                    'object': pre_entity, 'object_type': pre_entity_type})
                        
            result.append(dic)
        return result



                
class GPFilterTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPFilterTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )

    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_head_labels, batch_tail_labels = item
        logits1, logits2 = model(batch_token_ids, batch_mask_ids)

        loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits1, mask_zero=True)
        loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits2, mask_zero=True)
        loss = sum([loss1, loss2]) / 2
        
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits1[::2],dim=-1), F.softmax(logits1[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits1[1::2],dim=-1), F.softmax(logits1[::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits2[::2],dim=-1), F.softmax(logits2[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits2[1::2],dim=-1), F.softmax(logits2[::2],dim=-1), reduction='sum')
            # ’/ 4 * self.args.rdrop_alpha‘三是公式里带的, '/ 2'是为了头尾求平均
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits1.shape[0] / 2
        
        loss.backward()

        p1, r1, f11 = self.cal_prf(logits1, batch_head_labels)
        p2, r2, f12 = self.cal_prf(logits2, batch_tail_labels)
        p = (p1 + p2) / 2 
        r = (r1 + r2) / 2
        f1 = (f11 + f12) / 2
        return loss.detach(), p, r, f1

    def evaluate(self, model):
        isPbar=True
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device

        preds = []
        golds = []

        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            batch_token_ids, batch_mask_ids, batch_head_labels, batch_tail_labels = item
            batch_token_ids, batch_mask_ids, batch_head_labels, batch_tail_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_head_labels.to(device), batch_tail_labels.to(device)
            logits1, logits2 = model(batch_token_ids, batch_mask_ids)
            logits1 = torch.sigmoid(logits1[0].data).cpu().numpy()
            logits2 = torch.sigmoid(logits2[0].data).cpu().numpy()
            logits1[:, [0, -1]] -= np.inf
            logits1[:, :, [0, -1]] -= np.inf
            logits2[:, [0, -1]] -= np.inf
            logits2[:, :, [0, -1]] -= np.inf
            for l, h, t in zip(*np.where(logits1 > 0.5)):
                preds.append((step,l,h,t,'head'))
            for l, h, t in zip(*np.where(logits2 > 0.5)):
                preds.append((step,l,h,t,'tail'))
            for l in range(len(batch_head_labels[0])):
                a = batch_head_labels[0][l][0]
                h = a[0]
                t = a[1]
                if h != 0 or t != 0:
                    golds.append((step,l,h,t,'head'))
            for l in range(len(batch_tail_labels[0])):
                a = batch_head_labels[0][l][0]
                h = a[0]
                t = a[1]
                if h != 0 or t != 0:
                    golds.append((step,l,h,t,'tail'))
            
        if len(set(preds)) == 0:
            P=0
        else:
            P = len(set(preds) & set(golds)) / len(set(preds))
        R = len(set(preds) & set(golds)) / len(set(golds))
        if P+R ==0 :
            F = 0
        else:
            F = (2 * P * R) / (P + R)
        print('\n一个EPOCH分数：\n')
        print(str(round(P*100,2))+'|'+str(round(R*100,2))+'|'+str(round(F*100,2))+'|'+'\n')
        return P,R,F

    def cal_prf(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
                
    def predict_filter(self):
        args = self.args
        logger = self.logger
        model = self.model
        data_processor = self.data_processor
#         schema = data_processor.schema
        schema = data_processor.predicate2id
        tokenizer = self.tokenizer
        device = args.device
#         num_examples = 4482
        id2predicate = data_processor.id2predicate
        model.to(device)
        model.eval()
        
        output_dir = os.path.join('./result_output', 'filter_ace','gpf-'+args.model_version+'__gpner-'+args.prev_model_1+'__gpner9-'+args.prev_model_2)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = output_dir + '/test.jsonl'
        read_dir = os.path.join('./result_output', 'merge_ace','gpner-'+args.prev_model_1+'__gpner9-'+args.prev_model_2, 'test.jsonl')
        
        with jsonlines.open(output_dir, mode='w') as f, jsonlines.open(read_dir, mode='r') as test_samples:
            num_examples = len(open(read_dir).readlines())
            logger.info("***** Running prediction filter *****")
            logger.info("Num samples %d", num_examples)
            logger.info(f"***** write predict file to {output_dir} *****")
            pbar = ProgressBar(n_total=num_examples, desc='Filtering')
            for step, data in enumerate(test_samples):
                pbar(step)
                text = data['text']
                encoder_txt = tokenizer.encode_plus(text, max_length=args.max_length)
                input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
                attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
                scores = model(input_ids, attention_mask)
                outputs = [o[0].data.cpu().numpy() for o in scores]

                dic = {'text': text, 'spo_list': []}

                for spo in data['spo_list']:
                    sub = spo['subject']
                    obj = spo['object']
                    relation_key = spo['subject_type'] + "_" + spo['predicate'] + '_' + spo['object_type']
                    if relation_key not in schema:
                        continue
                    p = schema[relation_key]
                    s = tokenizer.encode(sub, add_special_tokens=False)
                    o = tokenizer.encode(obj, add_special_tokens=False)
                    sh = data_processor.search(s, encoder_txt["input_ids"])
                    oh = data_processor.search(o, encoder_txt["input_ids"])
                    if sh == -1:
                        s = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                        sh = self.data_processor.search(s, encoder_txt["input_ids"])
                    if oh == -1:
                        o = self.tokenizer.encode(' '+obj, add_special_tokens=False)
                        oh = self.data_processor.search(o, encoder_txt["input_ids"])
                    if sh != -1 and oh != -1:
                        st = sh + len(s) - 1
                        ot = oh + len(o) - 1
                        # 之前的预测结果不带 prob 字段，因此代码需要兼容
                        # and self.data_processor.regular(spo):
                        if (outputs[0][p, sh, oh] > args.filter_head_threshold and outputs[1][p, st, ot] > args.filter_tail_threshold):
                            if 'prob' in spo.keys():
                                del spo['prob']
                            dic['spo_list'].append(spo)
                        if 'prob' in spo.keys() and spo['prob'] > args.predict_threshold:
                            dic['spo_list'].append(spo)
                # 去重
                filter_set = set(ACESPO(spo,with_type=True) for spo in dic['spo_list'])
                dic['spo_list'] = []
                for spo in filter_set:
                    dic['spo_list'].append(spo.spo)
                f.write(dic)