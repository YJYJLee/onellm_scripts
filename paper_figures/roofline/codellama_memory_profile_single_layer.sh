# dram__throughput.avg.pct_of_peak_sustained_elapsed: Device memory utilization
# dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed: Device memory bandwidth utilization
# sm__throughput.avg.pct_of_peak_sustained_elapsed: SM utilization

# METRICS="gpu__dram_throughput.avg,gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__throughput.avg,gpu__time_active.sum"
# METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed"
# METRICS="dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS="dram__bytes_read.sum,dram__bytes_read.sum.per_second,sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dadd_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hadd_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_hmul_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum.per_second,smsp__cycles_elapsed.sum"

NVTX_COMMAND=" --nvtx-include "hello/""

CODELLAMA_34B=/fsx-atom/melhoushi/gpt_fast_ckpts/meta-llama/CodeLlama-34b-hf

dcgmi profile --pause

BATCH_SIZE=$1
PHASE=$2
SAVE_DIR="$HOME/codellama_result/humaneval_single_layer/batch_size${BATCH_SIZE}/$PHASE"
mkdir -p $SAVE_DIR
PHASE=$PHASE time /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f python /fsx-atom/yejinlee/2024_paper_projects/temp2/ATOM/eval_bigcode.py --checkpoint_path $CODELLAMA_34B/model.pth --model_name CodeLlama-34B --task humaneval --temperature 0.2 --top_p 0.95  --max_seq_len 512 --batch_size $BATCH_SIZE --log_dir ./logs/evals/codellama34b-baseline-humaneval-eager/ > $SAVE_DIR/stat.txt

/usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --metrics $METRICS --page raw  --csv --import $SAVE_DIR/ncu_memory_bw_profile.ncu-rep | awk -F ',' '{print $(NF-20) "\t" $(NF-19) "\t" $(NF-18) "\t" $(NF-17) "\t" $(NF-16) "\t" $(NF-15) "\t" $(NF-14) "\t" $(NF-13) "\t" $(NF-12) "\t" $(NF-11) "\t" $(NF-10) "\t" $(NF-9) "\t" $(NF-8) "\t" $(NF-7) "\t" $(NF-6) "\t" $(NF-5) "\t" $(NF-4) "\t" $(NF-3) "\t" $(NF-2) "\t"  $(NF-1) "\t" $NF}'  | awk -F '"' '{print $(NF-41) "\t" $(NF-39) "\t" $(NF-37) "\t" $(NF-35) "\t" $(NF-33) "\t" $(NF-31) "\t" $(NF-29) "\t" $(NF-27) "\t" $(NF-25) "\t" $(NF-23) "\t" $(NF-21) "\t" $(NF-19) "\t" $(NF-17) "\t" $(NF-15) "\t" $(NF-13) "\t" $(NF-11) "\t" $(NF-9) "\t" $(NF-7) "\t" $(NF-5) "\t" $(NF-3) "\t" $(NF-1)}'
