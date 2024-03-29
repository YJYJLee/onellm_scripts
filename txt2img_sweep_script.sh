TOME_R=(0.25 0.5 0.75 0.95)
TOME_LAYER=([0,1,2,3,4,5,6,7] [8,9,10,11,12,13,14,15] [16,17,18,19,20,21,22,23] [24,25,26,27,28,29,30,31] [0,1,2,3,4,5,6,7,8,9,10,11] [8,9,10,11,12,13,14,15,16,17,18,19] [12,13,14,15,16,17,18,19,20,21,22,23] [16,17,18,19,20,21,22,23,24,25,26,27] [20,21,22,23,24,25,26,27,28,29,30,31] [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] [12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27] [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
TOME_NUM_TOK=(34)
TOME_MERGE_TIME=(True False)

# TOME_LAYER=([0,1,2,3] [4,5,6,7] [8,9,10,11] [12,13,14,15] [14,15,16,17] [16,17,18,19] [20,21,22,23] [24,25,26,27] [28,29,30,31])
# TOME_NUM_TOK=(34 64 128 256 384 512 640 768 896)
# TOME_MERGE_TIME=(False)
BATCH_SIZE=32
NUM_SAMPLE=6    # first sample for warmup and use rest of it for the result


EXP_NAME=text_to_img_coco_image
MODEL_NAME=cm3v21_109m_sft
TASK=coco_image.0_shot
BASEBASE_DIR=/fsx-checkpoints/yejinlee/sweep/txt_to_img/${TASK}.${MODEL_NAME}/bs${BATCH_SIZE}
LOG_FILE=${BASEBASE_DIR}/sweep.txt

CFG=6
TEMP=1.0
TOPP=0.9
SEED=1

# Original
BASE_DIR=${BASEBASE_DIR}/original
EXP_DIR=${BASE_DIR}/$EXP_NAME/mn.${MODEL_NAME}.t.${TASK}.usecfg.True.cfg.${CFG}.temp.${TEMP}.topp.${TOPP}.seed.${SEED}

if [ -d "$EXP_DIR/images" ] ; then
    echo "`date` Skipping Generation... ${BASE_DIR}" >> $LOG_FILE
else
    echo "`date` [Generation] Start ${BASE_DIR}" >> $LOG_FILE
    mkdir -p $BASE_DIR
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name $EXP_NAME \
        --model-name $MODEL_NAME --batch-size $NUM_SAMPLE --tasks $TASK \
        --exp_dir $BASE_DIR --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCH_SIZE --generate > $BASE_DIR/time.txt
    echo "`date` [Generation] Finished!!! ${BASE_DIR}" >> $LOG_FILE
fi


if [[ (( -d "$EXP_DIR/raw_results" )) && (( -d "$EXP_DIR/results" )) ]] ; then
    echo "`date` Skipping Scoring... ${BASE_DIR}" >> $LOG_FILE
else
    echo "`date` [Scoring] Start ${BASE_DIR}" >> $LOG_FILE
    mv $EXP_DIR/images/chunk_1.jsonl $EXP_DIR/images/chunk_0.jsonl
    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name $EXP_NAME \
        --model-name $MODEL_NAME --batch-size $NUM_SAMPLE --tasks $TASK \
        --exp_dir $BASE_DIR --cfg $CFG --temp  $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCH_SIZE --score > $BASE_DIR/score.txt
    echo "`date` [Scording] finished!!! ${BASE_DIR}" >> $LOG_FILE
fi


# ToMe
for i in "${TOME_R[@]}"
do
    for j in "${TOME_LAYER[@]}"
    do
        for k in "${TOME_NUM_TOK[@]}"
        do
            for m in "${TOME_MERGE_TIME[@]}"
            do
                BASE_DIR=${BASEBASE_DIR}/r_${i}_layer_${j}_num_tok_${k}_merge_time_${m}
                mkdir -p $BASE_DIR
                EXP_DIR=${BASE_DIR}/${EXP_NAME}/mn.${MODEL_NAME}.t.${TASK}.usecfg.True.cfg.${CFG}.temp.${TEMP}.topp.${TOPP}.seed.${SEED}

                if [ -d "$EXP_DIR/images" ] ; then
                    echo "`date` Skipping Generation... ${BASE_DIR}" >> $LOG_FILE
                else
                    tome_measure_flag=""
                    if [ $m == "True" ] ; then
                        tome_measure_flag="--tome_merge_time"
                    fi

                    echo "`date` [Generation] Start ${BASE_DIR}" >> $LOG_FILE
                    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name $EXP_NAME \
                        --model-name $MODEL_NAME --batch-size $NUM_SAMPLE --tasks $TASK \
                        --exp_dir $BASE_DIR --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCH_SIZE --generate \
                        --tome True --tome_layer $j --tome_r $i --tome_num_tok $k $tome_measure_flag> $BASE_DIR/time.txt
                    echo "`date` [Generation] Finished!!! ${BASE_DIR}" >> $LOG_FILE
                fi


                if [[ (( -d "$EXP_DIR/raw_results" )) && (( -d "$EXP_DIR/results" )) ]] ; then
                    echo "`date` Skipping Scoring... ${BASE_DIR}" >> $LOG_FILE
                else
                    echo "`date` [Scoring] Start ${BASE_DIR}" >> $LOG_FILE
                    mv $EXP_DIR/images/chunk_1.jsonl $EXP_DIR/images/chunk_0.jsonl
                    FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok python scripts/cm3v2/gen_images.py --exp_name $EXP_NAME \
                        --model-name $MODEL_NAME --batch-size $NUM_SAMPLE --tasks $TASK \
                        --exp_dir $BASE_DIR --cfg $CFG --temp  $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCH_SIZE --score > $BASE_DIR/score.txt
                    echo "`date` [Scording] finished!!! ${BASE_DIR}" >> $LOG_FILE
                fi
            done
        done
    done
done