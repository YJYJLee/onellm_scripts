TOME_R=(0.25 0.5 0.75 0.95)
TOME_LAYER=("0 1 2 3" "4 5 6 7" "8 9 10 11" "12 13 14 15" "14 15 16 17" "16 17 18 19" "20 21 22 23" "24 25 26 27" "28 29 30 31" "0 1 2 3 4 5 6 7" "8 9 10 11 12 13 14 15" "16 17 18 19 20 21 22 23" "24 25 26 27 28 29 30 31" "0 1 2 3 4 5 6 7 8 9 10 11" "8 9 10 11 12 13 14 15 16 17 18 19" "12 13 14 15 16 17 18 19 20 21 22 23" "16 17 18 19 20 21 22 23 24 25 26 27" "20 21 22 23 24 25 26 27 28 29 30 31" "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"  "12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27" "16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31")
TOME_NUM_TOK=(34 64 128 256 384 512 640 768 896)
# TOME_NUM_TOK=(773)
TOME_MERGE_TIME=("True" "False")
# TOME_MERGE_TIME=(False)
BATCH_SIZE=32

MODEL_NAME=cm3v21_109m_sft
TASK=coco.0_shot.cm3v2_template
BASE_DIR=/fsx-checkpoints/yejinlee/sweep/img_to_txt/${TASK}.${MODEL_NAME}/bs${BATCH_SIZE}
LOG_FILE=$BASE_DIR/img2txt_sweep.txt

# Original
EXP_DIR=${BASE_DIR}/original

if [ -d "$EXP_DIR/results" ] ; then
    echo "`date` Skipping Generation... ${EXP_DIR}" >> $LOG_FILE
else
    mkdir -p $EXP_DIR
    echo "`date` [Generation] Start ${EXP_DIR}" >> $LOG_FILE
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py \
        --model-name $MODEL_NAME --tasks $TASK \
        --max-batch-size $BATCH_SIZE --dump-dir $EXP_DIR \
        --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" > $EXP_DIR/time.txt
    echo "`date` [Generation] Finished!!! ${EXP_DIR}" >> $LOG_FILE
fi

for i in "${TOME_R[@]}"
do
    for j in "${TOME_LAYER[@]}"
    do
        for k in "${TOME_NUM_TOK[@]}"
        do
            for m in "${TOME_MERGE_TIME[@]}"
            do
                layer_arr=(${j// /,})
                EXP_DIR=${BASE_DIR}/r_${i}_layer_[${layer_arr}]_num_tok_${k}_merge_time_${m}
                mkdir -p $EXP_DIR

                if [ -d "$EXP_DIR/results" ] ; then
                    echo "`date` Skipping Generation... ${EXP_DIR}" >> $LOG_FILE
                else
                    tome_measure_flag=""
                    if [ $m == "True" ] ; then
                        tome_measure_flag="--tome_merge_time"
                    fi
                    echo "`date` [Generation] Start ${EXP_DIR}" >> $LOG_FILE

                    echo "FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py \
                        --model-name $MODEL_NAME --tasks $TASK \
                        --max-batch-size $BATCH_SIZE --dump-dir $EXP_DIR \
                        --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" \
                        --tome --tome_layer ${j[@]} --tome_r $i --tome_num_tok $k $tome_measure_flag > $EXP_DIR/time.txt"
                    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python onellm_eval/run.py \
                        --model-name $MODEL_NAME --tasks $TASK \
                        --max-batch-size $BATCH_SIZE --dump-dir $EXP_DIR \
                        --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" \
                        --tome --tome_layer ${j[@]} --tome_r $i --tome_num_tok $k $tome_measure_flag > $EXP_DIR/time.txt


                    echo "`date` [Generation] Finished!!! ${EXP_DIR}" >> $LOG_FILE
                fi
            done
        done
    done
done
