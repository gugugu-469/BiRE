echo "pid:$$"
gpu=1
cd ../

data_dir=./ACE05-DyGIE/processed_data
echo "计算prf"

final_out_dir = ''
python get_prf.py --data_dir ${data_dir} --read_dir ${final_out_dir} --with_type
