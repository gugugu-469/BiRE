echo "pid:$$"
gpu=1
cd ../

data_dir=./ACE05-DyGIE/processed_data
method_name=ace05

model_type=bert
pretrained_model_name=chinese-bert-base


version1=01-21-13-45
version2=01-21-13-45
version3=01-21-13-46
echo "version1:${version1}"
python -u pyrun_gpnerace05.py --data_dir ${data_dir} --method_name ${method_name} --model_version ${version1}  --devices ${gpu} --model_type ${model_type} --pretrained_model_name ${pretrained_model_name} --learning_rate 4e-5 --do_predict --finetuned_model_name gpner


echo "version2:${version2}"
python -u pyrun_gpnerace05.py --data_dir ${data_dir} --method_name ${method_name} --model_version ${version2}  --devices ${gpu} --model_type ${model_type} --pretrained_model_name ${pretrained_model_name} --learning_rate 4e-5 --do_predict --finetuned_model_name gpner9


echo "结果融合_并集"
python 结果融合.py --prev_model_1 ${version1} --prev_model_2 ${version2} --type type1

echo "version3:${version3}"
python -u pyrun_gpfilter_ace05.py --data_dir ${data_dir} --method_name ${method_name} --model_version ${version3} --prev_model_1 ${version1} --prev_model_2 ${version2} --devices ${gpu} --model_type ${model_type} --pretrained_model_name ${pretrained_model_name} --learning_rate 4e-5 --do_filter --filter_head_threshold -5 --filter_tail_threshold -5

