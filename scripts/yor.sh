for j in 1 2 3 4 5
do
	export LANG=yo
	export DATA_DIR=data/nolly_${LANG}
	CUDA_VISIBLE_DEVICES=3 python run_textclass.py \
	  --model_name_or_path castorini/afriberta_large \
	  --do_train \
	  --do_eval \
	  --do_predict \
	  --per_device_train_batch_size 32 \
	  --learning_rate 5e-5 \
	  --num_train_epochs 20.0 \
	  --max_seq_length 200 \
	  --data_dir $DATA_DIR \
	  --output_dir ${LANG}_afriberta \
	  --save_steps -1 \
	  --overwrite_output_dir
	  
	  
	  CUDA_VISIBLE_DEVICES=3 python run_textclass.py \
	  --model_name_or_path xlm-roberta-large \
	  --do_train \
	  --do_eval \
	  --do_predict \
	  --per_device_train_batch_size 32 \
	  --learning_rate 5e-5 \
	  --num_train_epochs 20.0 \
	  --max_seq_length 200 \
	  --data_dir data/nolly_yo \
	  --output_dir ${LANG}_xlmrlarge \
	  --save_steps -1 \
	  --overwrite_output_dir
	  
	  
	  CUDA_VISIBLE_DEVICES=3 python run_textclass.py \
	  --model_name_or_path microsoft/mdeberta-v3-base \
	  --do_train \
	  --do_eval \
	  --do_predict \
	  --per_device_train_batch_size 32 \
	  --learning_rate 5e-5 \
	  --num_train_epochs 20.0 \
	  --max_seq_length 200 \
	  --data_dir data/nolly_yo \
	  --output_dir ${LANG}_mdeberta \
	  --save_steps -1 \
	  --overwrite_output_dir
	  
	  
	  CUDA_VISIBLE_DEVICES=3 python run_textclass.py \
	  --model_name_or_path Davlan/afro-xlmr-base \
	  --do_train \
	  --do_eval \
	  --do_predict \
	  --per_device_train_batch_size 32 \
	  --learning_rate 5e-5 \
	  --num_train_epochs 20.0 \
	  --max_seq_length 200 \
	  --data_dir data/yor \
	  --output_dir ${LANG}_afroxlmrbase \
	  --save_steps -1 \
	  --overwrite_output_dir
	  
	  
	  CUDA_VISIBLE_DEVICES=3 python run_textclass.py \
	  --model_name_or_path Davlan/afro-xlmr-large \
	  --do_train \
	  --do_eval \
	  --do_predict \
	  --per_device_train_batch_size 32 \
	  --learning_rate 5e-5 \
	  --num_train_epochs 20.0 \
	  --max_seq_length 200 \
	  --data_dir data/yor \
	  --output_dir ${LANG}_afroxlmrlarge \
	  --save_steps -1 \
	  --overwrite_output_dir
	  
	  
done

