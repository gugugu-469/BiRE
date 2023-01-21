import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from constant import text_start, left_bracket, right_bracket

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

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)
    
class GPNERDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train'
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = args.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        attention_mask = encoder_text["attention_mask"]
        spoes = set()

        for sub, sub_type in item["entity_list"]:
            # print(sub)
            sub_tokens = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub_tokens, input_ids)

            if sh != -1:
                spoes.add((sh, sh+len(sub_tokens)-1, class2id[sub_type]))
            else:
                sub_tokens = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                sh = self.data_processor.search(sub_tokens, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub_tokens)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        

        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return head_labels, input_ids, attention_mask

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids = [], []
        batch_head_labels = []
        for item in examples:
            head_labels, input_ids, attention_mask = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return batch_token_ids, batch_mask_ids, batch_head_labels


class GPFilterDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train'
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.predicate2id = data_processor.predicate2id #spo
        self.schema = data_processor.schema #spo
        self.args = args
        self.args.schema = self.schema
        
    def __len__(self):
        return len(self.samples)

    def encoder(self, item):
        args = self.args
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
#         token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o, s_t, o_t in item["spo_list"]:
            sub_tokens = self.tokenizer.encode(s, add_special_tokens=False)
            key = s_t + "_" + p + "_" +o_t
            p = self.predicate2id[key]
            obj_tokens = self.tokenizer.encode(o, add_special_tokens=False)
            sh = self.data_processor.search(sub_tokens, input_ids)
            oh = self.data_processor.search(obj_tokens, input_ids)
            
            if sh == -1:
                sub_tokens = self.tokenizer.encode(' '+s, add_special_tokens=False)
                sh = self.data_processor.search(sub_tokens, input_ids)
            if oh == -1:
                obj_tokens = self.tokenizer.encode(' '+o, add_special_tokens=False)
                oh = self.data_processor.search(obj_tokens, input_ids)
            if sh != -1 and oh != -1:
                spoes.add((sh, sh+len(sub_tokens)-1, p, oh, oh+len(obj_tokens)-1))

        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        for sh, st, p, oh, ot in spoes:
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        return head_labels, tail_labels, input_ids, attention_mask

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids = [], []
        batch_head_labels, batch_tail_labels = [], []
        for item in examples:
            head_labels, tail_labels, input_ids, attention_mask = item
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()

        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\

        return batch_token_ids, batch_mask_ids, batch_head_labels, batch_tail_labels