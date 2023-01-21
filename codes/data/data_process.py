import os
import json
import jsonlines
from collections import defaultdict
from constant import spot_labels, spot_prompt, asoc_prompt
from utils import random


class GPNERDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, 'train.json')
        self.dev_path = os.path.join(root, 'dev.json')
        self.test_path = os.path.join(root, 'test.json')
        self.schema_path = os.path.join(root, 'schemas.json')
        self._load_schema()
        
    def get_train_sample(self):
        return self._pre_process(self.train_path)

    def get_dev_sample(self):
        return self._pre_process(self.dev_path)

    def get_dev_sample_for_eval(self):
        with jsonlines.open(self.dev_path, 'r') as f:
            data_list = [line for line in f]
        return data_list

    def get_test_sample(self):
        with jsonlines.open(self.test_path, 'r') as f:
            data_list = [line for line in f]
        return data_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            labels = set()
            predicates = set()
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                labels.add(item['subject_type'])
                labels.add(item['object_type'])
                predicates.add(item['predicate'])
        labels = list(labels)
        predicates = list(predicates)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        self.labels = labels
        self.predicates = predicates
        self.num_predicates = len(predicates)
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        self.id2predicate = {i: v for i, v in enumerate(predicates)}
        self.args.num_labels = len(labels)
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def add_prefix(self, text, entity, predicate):
        return f"entity: {entity}, relation: {predicate}, {text}"
#         return f"{entity}[unused1]{predicate}[unused2]{text}"

    def build_negative_data(self, text, spo_list, entity2predicate_dic, data_type=1, keep_rate=1):
        args = self.args
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                temp = (spo['object'], spo['object_type'])
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append(temp)
                input_entity_types.append(spo["subject"])
            else:
                temp = (spo['subject'], spo['subject_type'])
                positive_dic[f"{spo['object']}{spo['predicate']}"].append(temp)
                input_entity_types.append(spo['object'])
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        
        for input_entity in input_entity_types:
            predicates = self.predicates
            for predicate in predicates:
                # 1：S+P抽O，2：O+P抽S
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # O+P 抽主体
                    "text": self.add_prefix(text, input_entity, predicate), # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                    "entity_list":[] # 必须是list
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                new_data.append(data)
        return new_data
    
    def _pre_process(self, path):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                new_data.append({
                    "type": 0, # 抽主体
                    "text": text,
                    "entity_list":[(spo["subject"], spo['subject_type']) for spo in spo_list] if self.args.finetuned_model_name == 'gpner' \
                                  else [(spo["object"], spo["object_type"]) for spo in spo_list]
                })
                if self.args.finetuned_model_name == 'gpner':
                    new_data.extend(self.build_negative_data(text, spo_list, self.subject_predicate_dic, 1, self.args.negative_samples_rate))
                else:
                    new_data.extend(self.build_negative_data(text, spo_list, self.object_predicate_dic, 2, self.args.negative_samples_rate))
        return new_data


class GPFilterDataProcessor(object):
    def __init__(self, args):
        self.args = args
        root = args.data_dir
        self.train_path = os.path.join(root, 'train.json')
        print(self.train_path)
        self.dev_path = os.path.join(root, 'dev.json')
        self.test_path = os.path.join(root, 'test.json')
        self.schema_path = os.path.join(root, 'schemas.json')
        self._load_schema()
        self.num_labels = len(self.predicate2id.keys())
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, mode='dev')

    def get_test_sample(self):
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip())["text"] for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            schema = []
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                schema.append(item["subject_type"]+"_"+item["predicate"]+"_"+item["object_type"])
        self.schema = schema
        self.num_predicates = len(schema)
        self.args.num_predicates = self.num_predicates
        self.predicate2id = {v: i for i, v in enumerate(schema)}
        self.id2predicate = {i: v for i, v in enumerate(schema)}
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with open(path, 'r', encoding='utf-8') as f:
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                line = json.loads(line)
                for _ in range(iter_num):
                    new_data.append({
                        "text":line["text"],
                        "spo_list":[(spo["subject"], spo["predicate"], spo["object"], spo["subject_type"], spo["object_type"]) for spo in line["spo_list"]]
                    })
        return new_data