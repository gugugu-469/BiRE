echo "pid:$$"
gpu=1
cd ../

data_dir=./ACE05-DyGIE/processed_data
method_name=ace05

model_type=bert
pretrained_model_name=chinese-bert-base


version1=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "version1:${version1}"
version1s[index]=${version1}
python -u pyrun_gpnerace05.py --data_dir ${data_dir} --method_name ${method_name} --model_version ${version1}  --devices ${gpu} --model_type ${model_type} --pretrained_model_name ${pretrained_model_name} --learning_rate 4e-5 --do_train --finetuned_model_name gpner --no_eval
echo "GPNER FINISH"


version2=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "version2:${version2}"
python -u pyrun_gpnerace05.py --data_dir ${data_dir} --method_name ${method_name} --model_version ${version2}  --devices ${gpu} --model_type ${model_type} --pretrained_model_name ${pretrained_model_name} --learning_rate 4e-5 --do_train --finetuned_model_name gpner9 --no_eval
echo "GPNER9 FINISH"

version3=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "version3:${version3}"
python -u pyrun_gpfilter_ace05.py --data_dir ${data_dir} --method_name ${method_name} --model_version ${version3}  --devices ${gpu} --model_type ${model_type} --pretrained_model_name ${pretrained_model_name} --learning_rate 4e-5 --do_train --no_eval
echo "GPFILTER FINISH"