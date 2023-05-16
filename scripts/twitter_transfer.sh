for j in 1 2 3 4 5
do	 
	export MAX_LENGTH=200
	export OUTPUT_PREDICTION=test_predictions
	export BATCH_SIZE=32
	export NUM_EPOCHS=20
	export SAVE_STEPS=500000
	export SEED=$j
	export LANG=ha
	export DATA_DIR=data/nolly_${LANG}
	export BERT_MODEL=castorini/afriberta_large
	export OUTPUT_DIR=twitter_${LANG}_afriberta$j
	export OUTPUT_FILE=test_result_${LANG}

	CUDA_VISIBLE_DEVICES=3 python3 train_textclass.py --data_dir $DATA_DIR \
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
	--do_predict \
	--overwrite_output_dir


	export LANG=ig
	export DATA_DIR=data/nolly_${LANG}
	export OUTPUT_FILE=test_result_${LANG}
	export OUTPUT_DIR=twitter_${LANG}_afriberta$j

	CUDA_VISIBLE_DEVICES=3 python3 train_textclass.py --data_dir $DATA_DIR \
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
	--do_predict \
	--overwrite_output_dir



	export LANG=pcm
	export DATA_DIR=data/nolly_${LANG}
	export OUTPUT_FILE=test_result_${LANG}
	export OUTPUT_DIR=twitter_${LANG}_afriberta$j

	CUDA_VISIBLE_DEVICES=3 python3 train_textclass.py --data_dir $DATA_DIR \
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
	--do_predict \
	--overwrite_output_dir
		


	export LANG=yo
	export DATA_DIR=data/nolly_${LANG}
	export OUTPUT_FILE=test_result_${LANG}
	export OUTPUT_DIR=twitter_${LANG}_afriberta$j

	CUDA_VISIBLE_DEVICES=3 python3 train_textclass.py --data_dir $DATA_DIR \
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
	--do_predict \
	--overwrite_output_dir

done




export LANG=pcm
export BERT_MODEL=twitter_${LANG}_afriberta1
export DATA_DIR=data/nolly_${LANG}
export OUTPUT_FILE=test_result_${LANG}
export OUTPUT_DIR=twitter_nolly_${LANG}

CUDA_VISIBLE_DEVICES=3 python3 train_textclass.py --data_dir $DATA_DIR \
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
--do_predict \
--overwrite_output_dir