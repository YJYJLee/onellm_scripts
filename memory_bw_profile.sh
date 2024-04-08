# dram__throughput.avg.pct_of_peak_sustained_elapsed: Device memory utilization
# dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed: Device memory bandwidth utilization
# sm__throughput.avg.pct_of_peak_sustained_elapsed: SM utilization

# METRICS="gpu__dram_throughput.avg,gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__throughput.avg,gpu__time_active.sum"
# METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS="dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed"
PROFILE_LAYER=5
# Run profile
MODELNAME="cm3v21_109m_sft"
BASEFOLDER="/fsx-checkpoints/yejinlee/cm3v2_memory_bw_profile"
BATCHSIZE=$1
PROFILE_STEP=$2

# Image to Text
## MSCOCO
TASK="coco.0_shot.cm3v2_template"
SAVE_DIR=$BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/layer${PROFILE_LAYER}
mkdir -p $SAVE_DIR
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx --nvtx-include "hello/" -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --memory_profile_step $PROFILE_STEP > $SAVE_DIR/stat.txt
## Flickr30k
TASK="flickr30k.0_shot.cm3v2_template"
SAVE_DIR=$BASEFOLDER/i2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/layer${PROFILE_LAYER}
mkdir -p $SAVE_DIR
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx --nvtx-include "hello/" -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}"  --memory_profile_step $PROFILE_STEP > $SAVE_DIR/stat.txt

# Text to Image
CFG=6
TEMP=1.0
TOPP=0.9
SEED=1
## COCO
TASK="coco_image.0_shot"
SAVE_DIR=$BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}/layer${PROFILE_LAYER}
mkdir -p $SAVE_DIR
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx --nvtx-include "hello/" -f python scripts/cm3v2/gen_images.py --exp_name text_to_img_coco_image --model-name $MODELNAME --batch-size 10 --tasks $TASK --exp_dir $SAVE_DIR --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate --memory_profile_step $PROFILE_STEP > $SAVE_DIR/stat.txt
## Partiprompts
TASK="partiprompts.0_shot"
mkdir -p $BASEFOLDER/t2i.${MODELNAME}.bs${BATCHSIZE}.${TASK}.cfg${CFG}.temp${TEMP}.topp${TOPP}.seed.${SEED}/layer${PROFILE_LAYER}
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx --nvtx-include "hello/" -f python scripts/cm3v2/gen_images.py --exp_name text_to_img_partiprompts --model-name $MODELNAME --batch-size 10 --tasks $TASK --exp_dir $SAVE_DIR --cfg $CFG --temp $TEMP --topp $TOPP --seed $SEED --num-cfg-samples $BATCHSIZE --generate --memory_profile_step $PROFILE_STEP > $SAVE_DIR/stat.txt

# Image+Text to Text
## OKVQA
TASK="okvqa.0_shot.cm3v2_template"
SAVE_DIR=$BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/layer${PROFILE_LAYER}
mkdir -p $SAVE_DIR
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx --nvtx-include "hello/" -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --memory_profile_step $PROFILE_STEP > $SAVE_DIR/stat.txt
## TextVQA
TASK="textvqa.0_shot.cm3v2_template"
SAVE_DIR=$BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/layer${PROFILE_LAYER}
mkdir -p $SAVE_DIR
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx --nvtx-include "hello/" -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --memory_profile_step $PROFILE_STEP > $SAVE_DIR/stat.txt
## Vizwiz
TASK="vizwiz.0_shot.cm3v2_template"
SAVE_DIR=$BASEFOLDER/it2t.${MODELNAME}.bs${BATCHSIZE}.${TASK}/layer${PROFILE_LAYER}
mkdir -p $SAVE_DIR
FSD=/fsx-onellm/mingdachen/onellm-eval-data-tok /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx --nvtx-include "hello/" -f python onellm_eval/run.py  --model-name $MODELNAME --tasks $TASK --max-batch-size $BATCHSIZE --dump-dir $SAVE_DIR --use-model-config-args --predictor-kwargs "{\"generation_mode\": \"text\"}" --memory_profile_step $PROFILE_STEP > $SAVE_DIR/stat.txt



# ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed --page raw  --csv --import ncu_memory_bw_profile.ncu-rep  | awk -F ',' '{print $(NF-3) "\t" $(NF-2) "\t"  $(NF-1),"\t",$NF}'
