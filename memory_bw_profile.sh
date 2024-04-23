# dram__throughput.avg.pct_of_peak_sustained_elapsed: Device memory utilization
# dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed: Device memory bandwidth utilization
# sm__throughput.avg.pct_of_peak_sustained_elapsed: SM utilization

# METRICS="gpu__dram_throughput.avg,gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__throughput.avg,gpu__time_active.sum"
# METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed"
PROFILE_LAYER=5
# Run profile
MODELNAME="cm3v21_109m_sft"
BASEFOLDER="/fsx-checkpoints/yejinlee/cm3v2_memory_bw_profile"
PROFILE_GRANULARITY=8
BATCHSIZE=$1

# Image to Text
## MSCOCO
TASK="coco.0_shot.cm3v2_template"
END=13
for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
do
    NVTX_COMMAND=""
    PROFILE_STEPS=""
    PROFILE_STEPS_STR=""
    for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
    do
        tmp=" --nvtx-include "hello_${ps}/""
        NVTX_COMMAND+=$tmp
        PROFILE_STEPS+="${ps} "
        PROFILE_STEPS_STR+="${ps}."
    done
    SAVE_DIR=$BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
    if [ -d "$SAVE_DIR/results" ] ; then
        echo "`date` Skipping Generation... ${SAVE_DIR}"
    else
        mkdir -p $SAVE_DIR
        time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
    fi
done


## Flickr30k
TASK="flickr30k.0_shot.cm3v2_template"
END=13
for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
do
    NVTX_COMMAND=""
    PROFILE_STEPS=""
    PROFILE_STEPS_STR=""
    for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
    do
        tmp=" --nvtx-include "hello_${ps}/""
        NVTX_COMMAND+=$tmp
        PROFILE_STEPS+="${ps} "
        PROFILE_STEPS_STR+="${ps}."
    done
    SAVE_DIR=$BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
    if [ -d "$SAVE_DIR/results" ] ; then
        echo "`date` Skipping Generation... ${SAVE_DIR}"
    else
        mkdir -p $SAVE_DIR
        time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}"  --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
    fi
done



# Text to Image
CFG=6
TEMP=1.0
TOPP=0.9
SEED=1
## COCO
TASK="coco_image.0_shot"
END=1024
for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
do
    NVTX_COMMAND=""
    PROFILE_STEPS="["
    PROFILE_STEPS_STR=""
    for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
    do
        tmp=" --nvtx-include "hello_${ps}/""
        NVTX_COMMAND+=$tmp
        PROFILE_STEPS+="${ps},"
        PROFILE_STEPS_STR+="${ps}."
    done
    PROFILE_STEPS+="]"
    SAVE_DIR=$BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg.${CFG}.temp.${TEMP}.topp.${TOPP}.seed.${SEED}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
    if [ -d "$SAVE_DIR/images" ] ; then
        echo "`date` Skipping Generation... ${SAVE_DIR}"
    else
        mkdir -p $SAVE_DIR
        time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python scripts/cm3v2/gen_images.py --exp_name text_to_img_coco_image --model-name $MODELNAME --batch-size 10 --tasks $TASK --exp_dir $SAVE_DIR --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
    fi
done


## Partiprompts
TASK="partiprompts.0_shot"
END=1024
for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
do
    NVTX_COMMAND=""
    PROFILE_STEPS="["
    PROFILE_STEPS_STR=""
    for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
    do
        tmp=" --nvtx-include "hello_${ps}/""
        NVTX_COMMAND+=$tmp
        PROFILE_STEPS+="${ps},"
        PROFILE_STEPS_STR+="${ps}."
    done
    PROFILE_STEPS+="]"
    SAVE_DIR=$BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg.${CFG}.temp.${TEMP}.topp.${TOPP}.seed.${SEED}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
    if [ -d "$SAVE_DIR/images" ] ; then
        echo "`date` Skipping Generation... ${SAVE_DIR}"
    else
        mkdir -p $SAVE_DIR
        time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python scripts/cm3v2/gen_images.py --exp_name text_to_img_partiprompts --model-name $MODELNAME --batch-size 10 --tasks $TASK --exp_dir $SAVE_DIR --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
    fi
done



# Image+Text to Text
## OKVQA
TASK="okvqa.0_shot.cm3v2_template"
END=4
for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
do
    NVTX_COMMAND=""
    PROFILE_STEPS=""
    PROFILE_STEPS_STR=""
    for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
    do
        tmp=" --nvtx-include "hello_${ps}/""
        NVTX_COMMAND+=$tmp
        PROFILE_STEPS+="${ps} "
        PROFILE_STEPS_STR+="${ps}."
    done
    SAVE_DIR=$BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
    if [ -d "$SAVE_DIR/results" ] ; then
        echo "`date` Skipping Generation... ${SAVE_DIR}"
    else
        mkdir -p $SAVE_DIR
        time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
    fi
done



## TextVQA
TASK="textvqa.0_shot.cm3v2_template"
END=8
for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
do
    NVTX_COMMAND=""
    PROFILE_STEPS=""
    PROFILE_STEPS_STR=""
    for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
    do
        tmp=" --nvtx-include "hello_${ps}/""
        NVTX_COMMAND+=$tmp
        PROFILE_STEPS+="${ps} "
        PROFILE_STEPS_STR+="${ps}."
    done
    SAVE_DIR=$BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
    if [ -d "$SAVE_DIR/results" ] ; then
        echo "`date` Skipping Generation... ${SAVE_DIR}"
    else
        mkdir -p $SAVE_DIR
        time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
    fi
done


## Vizwiz
TASK="vizwiz.0_shot.cm3v2_template"
END=5
for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
do
    NVTX_COMMAND=""
    PROFILE_STEPS=""
    PROFILE_STEPS_STR=""
    for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
    do
        tmp=" --nvtx-include "hello_${ps}/""
        NVTX_COMMAND+=$tmp
        PROFILE_STEPS+="${ps} "
        PROFILE_STEPS_STR+="${ps}."
    done
    SAVE_DIR=$BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
    if [ -d "$SAVE_DIR/results" ] ; then
        echo "`date` Skipping Generation... ${SAVE_DIR}"
    else
        mkdir -p $SAVE_DIR
        time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
    fi
done


# /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed --page raw  --csv --import /fsx-checkpoints/yejinlee/cm3v2_memory_bw_profile/i2t.cm3v21_109m_sft.bs1.coco.0_shot.cm3v2_template/layer5/decoding_step0/ncu_memory_bw_profile.ncu-rep  | awk -F ',' '{print $(NF-5) "\t" $(NF-4) "\t" $(NF-3) "\t" $(NF-2) "\t"  $(NF-1) "\t" $NF}' | tail -n +2 | awk -F '"' '{print $(NF-11) "\t" $(NF-9) "\t" $(NF-7) "\t" $(NF-5) "\t" $(NF-3) "\t" $(NF-1)}'
