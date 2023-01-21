
import jsonlines
import os
from codes.utils import SPO, ACESPO
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import shutil

# ## ace取并集

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--read_path_1', type=str, default='',
                        help="whether to use entity type information")
    
    parser.add_argument('--read_path_2', type=str, default='',
                        help="whether to use entity type information")

    parser.add_argument('--out_path', type=str, default='',
                        help="whether to use entity type information")

    parser.add_argument('--type', type=str, default='type1',
                        help="type1为并集，其余为交集")

    parser.add_argument('--filter_head_threshold', type=float, default=-999.,
                        help="whether to use entity type information")

    parser.add_argument('--filter_tail_threshold', type=float, default=-999.,
                        help="whether to use entity type information")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    

    if args.type=='type1':
        read1 = args.read_path_1
        read2 = args.read_path_2
        output_dir = args.out_path
        if args.filter_head_threshold != -999.:
            if 'json' not in read1:
                read1 = read1+'_'+str(args.filter_head_threshold)+'_'+str(args.filter_tail_threshold)+'.jsonl'
            if 'json' not in read2:
                read2 = read2+'_'+str(args.filter_head_threshold)+'_'+str(args.filter_tail_threshold)+'.jsonl'
            if 'json' not in output_dir:
                output_dir = output_dir+'_'+str(args.filter_head_threshold)+'_'+str(args.filter_tail_threshold)+'.jsonl'

        with jsonlines.open(f'{output_dir}', mode='w') as w:
            with jsonlines.open(read1, mode='r') as r1, jsonlines.open(read2, mode='r') as r2:
                for data1, data2 in zip(r1, r2):
                    dic = {'text': data1['text'], 'spo_list': data1['spo_list'] + data2['spo_list']}
                    w.write(dic)
    else:
        read1 = args.read_path_1
        read2 = args.read_path_2
        output_dir = args.out_path

        with jsonlines.open(f'{output_dir}', mode='w') as w:
            with jsonlines.open(read1, mode='r') as r1, jsonlines.open(read2, mode='r') as r2:
                for data1, data2 in zip(r1, r2):
                    dic = {'text': data1['text'], 'spo_list': []}
                    for spo in (set(ACESPO(spo) for spo in data1['spo_list']) & set(ACESPO(spo) for spo in data2['spo_list'])):
                        dic['spo_list'].append(spo.spo)
                    w.write(dic)


if __name__ == '__main__':
    main()

