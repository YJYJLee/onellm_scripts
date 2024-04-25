# dram__throughput.avg.pct_of_peak_sustained_elapsed: Device memory utilization
# dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed: Device memory bandwidth utilization
# sm__throughput.avg.pct_of_peak_sustained_elapsed: SM utilization

# METRICS="gpu__dram_throughput.avg,gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__throughput.avg,gpu__time_active.sum"
# METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed"
PROFILE_LAYER=5
# Run profile
MODELNAME="cm3v21_109m_sft"
BASEFOLDER="/fsx-checkpoints/yejinlee/cm3v2_memory_bw_profile/wo_efficient_attn/retrieval"
PROFILE_GRANULARITY=8
BATCHSIZE=$1
NRETRIEVED_DOCS=$2

# Image to Text
## MSCOCO
TASK="coco.0_shot.flamingo_retrieval_v2_template"

if [ $BATCHSIZE -eq 1 ]
then
    END=30
elif [ $BATCHSIZE -eq 4 ]
then
    if [ $NRETRIEVED_DOCS -eq 1 ]
    then
        END=34
    elif [ $NRETRIEVED_DOCS -eq 2 ]
    then
        END=49
    elif [ $NRETRIEVED_DOCS -eq 3 ]
    then
        END=30
    elif [ $NRETRIEVED_DOCS -eq 4 ]
    then
        END=30
    fi
elif [ $BATCHSIZE -eq 8 ]
then
    if [ $NRETRIEVED_DOCS -eq 1 ]
    then
        END=34
    elif [ $NRETRIEVED_DOCS -eq 2 ]
    then
        END=40
    elif [ $NRETRIEVED_DOCS -eq 3 ]
    then
        END=30
    elif [ $NRETRIEVED_DOCS -eq 4 ]
    then
        END=30
    fi
elif [ $BATCHSIZE -eq 16 ]
then
    if [ $NRETRIEVED_DOCS -eq 1 ]
    then
        END=39
    elif [ $NRETRIEVED_DOCS -eq 2 ]
    then
        END=45
    elif [ $NRETRIEVED_DOCS -eq 3 ]
    then
        END=30
    elif [ $NRETRIEVED_DOCS -eq 4 ]
    then
        END=30
    fi
fi


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
    SAVE_DIR=$BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
    if [ -d "$SAVE_DIR/results" ] ; then
        echo "`date` Skipping Generation... ${SAVE_DIR}"
    else
        mkdir -p $SAVE_DIR
        time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --n-retrieved-docs $NRETRIEVED_DOCS --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
                                                                                                                                                                                                                                                            
    fi
done



# ## Flickr30k
# TASK="flickr30k.0_shot.flamingo_retrieval_v2_template"
# if [ $BATCHSIZE -eq 1 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=30
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=30
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=30
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=29
#     fi
# elif [ $BATCHSIZE -eq 4 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=37
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=46
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=30
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=30
#     fi
# elif [ $BATCHSIZE -eq 8 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=38
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=40
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=30
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=30
#     fi
# elif [ $BATCHSIZE -eq 16 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=43
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=52
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=30
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=30
#     fi
# fi

# for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
# do
#     NVTX_COMMAND=""
#     PROFILE_STEPS=""
#     PROFILE_STEPS_STR=""
#     for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
#     do
#         tmp=" --nvtx-include "hello_${ps}/""
#         NVTX_COMMAND+=$tmp
#         PROFILE_STEPS+="${ps} "
#         PROFILE_STEPS_STR+="${ps}."
#     done
#     SAVE_DIR=$BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
#     if [ -d "$SAVE_DIR/results" ] ; then
#         echo "`date` Skipping Generation... ${SAVE_DIR}"
#     else
#         mkdir -p $SAVE_DIR
#         time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}"  --n-retrieved-docs $NRETRIEVED_DOCS --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
#     fi
# done



# # Text to Image
# CFG=6
# TEMP=1.0
# TOPP=0.9
# SEED=1
# ## COCO
# TASK="coco_image.0_shot.default_retrieval_template"
# END=1024
# for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
# do
#     NVTX_COMMAND=""
#     PROFILE_STEPS="["
#     PROFILE_STEPS_STR=""
#     for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
#     do
#         tmp=" --nvtx-include "hello_${ps}/""
#         NVTX_COMMAND+=$tmp
#         PROFILE_STEPS+="${ps},"
#         PROFILE_STEPS_STR+="${ps}."
#     done
#     PROFILE_STEPS+="]"
#     SAVE_DIR=$BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg.${CFG}.temp.${TEMP}.topp.${TOPP}.seed.${SEED}.n_retrieved_docs${NRETRIEVED_DOCS}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
#     if [ -d "$SAVE_DIR/images" ] ; then
#         echo "`date` Skipping Generation... ${SAVE_DIR}"
#     else
#         mkdir -p $SAVE_DIR
#         time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python scripts/cm3v2/gen_images.py --exp_name text_to_img_coco_image --model-name $MODELNAME --batch-size 10 --tasks $TASK --exp_dir $SAVE_DIR --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate --n-retrieved-docs $NRETRIEVED_DOCS --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
#     fi
# done


# ## Partiprompts
# TASK="partiprompts.0_shot.default_retrieval_template"
# END=1024
# for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
# do
#     NVTX_COMMAND=""
#     PROFILE_STEPS="["
#     PROFILE_STEPS_STR=""
#     for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
#     do
#         tmp=" --nvtx-include "hello_${ps}/""
#         NVTX_COMMAND+=$tmp
#         PROFILE_STEPS+="${ps},"
#         PROFILE_STEPS_STR+="${ps}."
#     done
#     PROFILE_STEPS+="]"
#     SAVE_DIR=$BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg.${CFG}.temp.${TEMP}.topp.${TOPP}.seed.${SEED}.n_retrieved_docs${NRETRIEVED_DOCS}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
#     if [ -d "$SAVE_DIR/images" ] ; then
#         echo "`date` Skipping Generation... ${SAVE_DIR}"
#     else
#         mkdir -p $SAVE_DIR
#         time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python scripts/cm3v2/gen_images.py --exp_name text_to_img_partiprompts --model-name $MODELNAME --batch-size 10 --tasks $TASK --exp_dir $SAVE_DIR --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate --n-retrieved-docs $NRETRIEVED_DOCS --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
#     fi
# done



# # Image+Text to Text
# ## OKVQA
# TASK="okvqa.0_shot.flamingo_retrieval_v2_template"
# if [ $BATCHSIZE -eq 1 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=10
#     fi
# elif [ $BATCHSIZE -eq 4 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=18
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=17
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=10
#     fi
# elif [ $BATCHSIZE -eq 8 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=19
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=21
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=10
#     fi
# elif [ $BATCHSIZE -eq 16 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=21
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=24
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=10
#     fi
# fi
# for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
# do
#     NVTX_COMMAND=""
#     PROFILE_STEPS=""
#     PROFILE_STEPS_STR=""
#     for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
#     do
#         tmp=" --nvtx-include "hello_${ps}/""
#         NVTX_COMMAND+=$tmp
#         PROFILE_STEPS+="${ps} "
#         PROFILE_STEPS_STR+="${ps}."
#     done
#     SAVE_DIR=$BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
#     if [ -d "$SAVE_DIR/results" ] ; then
#         echo "`date` Skipping Generation... ${SAVE_DIR}"
#     else
#         mkdir -p $SAVE_DIR
#         time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --n-retrieved-docs $NRETRIEVED_DOCS --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
#     fi
# done



# ## TextVQA
# TASK="textvqa.0_shot.flamingo_retrieval_v2_template"
# if [ $BATCHSIZE -eq 1 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=10
#     fi
# elif [ $BATCHSIZE -eq 4 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=12
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=20
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=10
#     fi
# elif [ $BATCHSIZE -eq 8 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=14
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=22
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=10
#     fi
# elif [ $BATCHSIZE -eq 16 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=21
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=32
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=10
#     fi
# fi
# for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
# do
#     NVTX_COMMAND=""
#     PROFILE_STEPS=""
#     PROFILE_STEPS_STR=""
#     for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
#     do
#         tmp=" --nvtx-include "hello_${ps}/""
#         NVTX_COMMAND+=$tmp
#         PROFILE_STEPS+="${ps} "
#         PROFILE_STEPS_STR+="${ps}."
#     done
#     SAVE_DIR=$BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
#     if [ -d "$SAVE_DIR/results" ] ; then
#         echo "`date` Skipping Generation... ${SAVE_DIR}"
#     else
#         mkdir -p $SAVE_DIR
#         time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --n-retrieved-docs $NRETRIEVED_DOCS --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
#     fi
# done


# ## Vizwiz
# TASK="vizwiz.0_shot.flamingo_retrieval_v2_template"
# if [ $BATCHSIZE -eq 1 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=10
#     fi
# elif [ $BATCHSIZE -eq 4 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=24
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=18
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=10
#     fi
# elif [ $BATCHSIZE -eq 8 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=14
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=19
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=10
#     fi
# elif [ $BATCHSIZE -eq 16 ]
# then
#     if [ $NRETRIEVED_DOCS -eq 1 ]
#     then
#         END=21
#     elif [ $NRETRIEVED_DOCS -eq 2 ]
#     then
#         END=26
#     elif [ $NRETRIEVED_DOCS -eq 3 ]
#     then
#         END=10
#     elif [ $NRETRIEVED_DOCS -eq 4 ]
#     then
#         END=10
#     fi
# fi
# for ((profile_step=0;profile_step<=END;profile_step+=$PROFILE_GRANULARITY));
# do
#     NVTX_COMMAND=""
#     PROFILE_STEPS=""
#     PROFILE_STEPS_STR=""
#     for ((ps=profile_step;ps<=END&&ps<profile_step+$PROFILE_GRANULARITY;ps++));
#     do
#         tmp=" --nvtx-include "hello_${ps}/""
#         NVTX_COMMAND+=$tmp
#         PROFILE_STEPS+="${ps} "
#         PROFILE_STEPS_STR+="${ps}."
#     done
#     SAVE_DIR=$BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}.n_retrieved_docs${NRETRIEVED_DOCS}/layer${PROFILE_LAYER}/decoding_step_${PROFILE_STEPS_STR}
#     if [ -d "$SAVE_DIR/results" ] ; then
#         echo "`date` Skipping Generation... ${SAVE_DIR}"
#     else
#         mkdir -p $SAVE_DIR
#         time FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --n-retrieved-docs $NRETRIEVED_DOCS --memory_profile_step $PROFILE_STEPS > $SAVE_DIR/stat.txt
#     fi
# done


# /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed --page raw  --csv --import /fsx-checkpoints/yejinlee/cm3v2_memory_bw_profile/i2t.cm3v21_109m_sft.bs1.coco.0_shot.cm3v2_template/layer5/decoding_step0/ncu_memory_bw_profile.ncu-rep  | awk -F ',' '{print $(NF-5) "\t" $(NF-4) "\t" $(NF-3) "\t" $(NF-2) "\t"  $(NF-1) "\t" $NF}' | tail -n +2 | awk -F '"' '{print $(NF-11) "\t" $(NF-9) "\t" $(NF-7) "\t" $(NF-5) "\t" $(NF-3) "\t" $(NF-1)}'
