export MAX_LENGTH=200
export BATCH_SIZE=32
export NUM_EPOCHS=20
export SAVE_STEPS=500000
export BERT_MODEL=Davlan/afro-xlmr-base


for j in 1 2 3 4 5
do
  for LANG in ha ig yo pcm
	do
    export OUTPUT_FILE=test_result$j
    export OUTPUT_PREDICTION=test_predictions$j
    export SEED=$j
    export DATA_DIR=data/nolly_${LANG}


    export OUTPUT_DIR=${LANG}_afroxlmrlarge

    CUDA_VISIBLE_DEVICES=0,1 python3 train_textclass.py --data_dir $DATA_DIR \
    --model_type xlmroberta \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --output_result $OUTPUT_FILE \
    --output_prediction_file $OUTPUT_PREDICTION \
    --max_seq_length  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate 1e-5 \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --do_train \
    --do_predict \
    --overwrite_output_dir
    done
  done