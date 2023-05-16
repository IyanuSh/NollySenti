for j in 1 2 3 4 5
do	  

	export MAX_LENGTH=200
	export OUTPUT_FILE=test_result$j
	export OUTPUT_PREDICTION=test_predictions
	export BATCH_SIZE=32
	export NUM_EPOCHS=20
	export SAVE_STEPS=500000
	export SEED=$j
	export LANG=pcm
	export DATA_DIR=data/nolly_${LANG}
	
	
	export BERT_MODEL=xlm-roberta-base
	export OUTPUT_DIR=${LANG}_xlmrbase

	CUDA_VISIBLE_DEVICES=2 python3 train_textclass.py --data_dir $DATA_DIR \
	--model_type xlmroberta \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--output_result $OUTPUT_FILE \
	--output_prediction_file $OUTPUT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--learning_rate 5e-5 \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--seed $SEED \
	--do_train \
	--do_predict \
	--overwrite_output_dir
	
	
	export BERT_MODEL=microsoft/mdeberta-v3-base
	export OUTPUT_DIR=${LANG}_mdeberta
	
	CUDA_VISIBLE_DEVICES=2 python3 train_textclass.py --data_dir $DATA_DIR \
	--model_type xlmroberta \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--output_result $OUTPUT_FILE \
	--output_prediction_file $OUTPUT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--learning_rate 5e-5 \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--seed $SEED \
	--do_train \
	--do_predict \
	--overwrite_output_dir
	
	
	
	export BERT_MODEL=Davlan/afro-xlmr-base
	export OUTPUT_DIR=${LANG}_afroxlmrbase

	CUDA_VISIBLE_DEVICES=2 python3 train_textclass.py --data_dir $DATA_DIR \
	--model_type xlmroberta \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--output_result $OUTPUT_FILE \
	--output_prediction_file $OUTPUT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--learning_rate 5e-5 \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--seed $SEED \
	--do_train \
	--do_predict \
	--overwrite_output_dir
	
	
	export BERT_MODEL=bert-base-multilingual-cased
	export OUTPUT_DIR=${LANG}_bert
	
	CUDA_VISIBLE_DEVICES=2 python3 train_textclass.py --data_dir $DATA_DIR \
	--model_type bert \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--output_result $OUTPUT_FILE \
	--output_prediction_file $OUTPUT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--learning_rate 5e-5 \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--seed $SEED \
	--do_train \
	--do_predict \
	--overwrite_output_dir
	
	
	export BERT_MODEL=castorini/afriberta_large
	export OUTPUT_DIR=${LANG}_afriberta

	CUDA_VISIBLE_DEVICES=2 python3 train_textclass.py --data_dir $DATA_DIR \
	--model_type xlmroberta \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--output_result $OUTPUT_FILE \
	--output_prediction_file $OUTPUT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--learning_rate 5e-5 \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--seed $SEED \
	--do_train \
	--do_predict \
	--overwrite_output_dir
	
	
	
	
	
	done
