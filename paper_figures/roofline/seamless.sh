# dram__throughput.avg.pct_of_peak_sustained_elapsed: Device memory utilization
# dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed: Device memory bandwidth utilization
# sm__throughput.avg.pct_of_peak_sustained_elapsed: SM utilization

# METRICS="gpu__dram_throughput.avg,gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__throughput.avg,gpu__time_active.sum"
# METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed"
# METRICS="dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS="dram__bytes_read.sum,dram__bytes_read.sum.per_second,sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dadd_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hadd_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_hmul_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum.per_second,smsp__cycles_elapsed.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,sm__cycles_active.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed"

dcgmi profile --pause

NVTX_COMMAND=" --nvtx-include "hello/""

SAVE_DIR="/home/yejinlee/roofline_profile/seamless/S2ST"
mkdir -p $SAVE_DIR
EFFECTIVE_BATCH_SIZE=$1 time /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f m4t_evaluate --data_file /fsx-atom/shared/yejinlee_backups/seamless_dataset2/test_manifest.json --task S2ST --tgt_lang spa --output_path ./test --ref_field tgt_text --audio_root_dir /fsx-atom/shared/yejinlee_backups/seamless_dataset2/downloads/extracted/7411e52500e75f4587cdf715b30663e27f8e5f4865174435569bffe9b3899945/test --model_name seamlessM4T_v2_large --batch_size 1 > $SAVE_DIR/stat.txt

/usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --metrics $METRICS --page raw  --csv --import $SAVE_DIR/ncu_memory_bw_profile.ncu-rep | awk -F ',' '{print $(NF-20) "\t" $(NF-19) "\t" $(NF-18) "\t" $(NF-17) "\t" $(NF-16) "\t" $(NF-15) "\t" $(NF-14) "\t" $(NF-13) "\t" $(NF-12) "\t" $(NF-11) "\t" $(NF-10) "\t" $(NF-9) "\t" $(NF-8) "\t" $(NF-7) "\t" $(NF-6) "\t" $(NF-5) "\t" $(NF-4) "\t" $(NF-3) "\t" $(NF-2) "\t"  $(NF-1) "\t" $NF}'  | awk -F '"' '{print $(NF-41) "\t" $(NF-39) "\t" $(NF-37) "\t" $(NF-35) "\t" $(NF-33) "\t" $(NF-31) "\t" $(NF-29) "\t" $(NF-27) "\t" $(NF-25) "\t" $(NF-23) "\t" $(NF-21) "\t" $(NF-19) "\t" $(NF-17) "\t" $(NF-15) "\t" $(NF-13) "\t" $(NF-11) "\t" $(NF-9) "\t" $(NF-7) "\t" $(NF-5) "\t" $(NF-3) "\t" $(NF-1)}'



SAVE_DIR="/home/yejinlee/roofline_profile/seamless/S2TT"
mkdir -p $SAVE_DIR
EFFECTIVE_BATCH_SIZE=$1 time /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f m4t_evaluate --data_file /fsx-atom/shared/yejinlee_backups/seamless_dataset2/test_manifest.json --task S2TT --tgt_lang spa --output_path ./test --ref_field tgt_text --audio_root_dir /fsx-atom/shared/yejinlee_backups/seamless_dataset2/downloads/extracted/7411e52500e75f4587cdf715b30663e27f8e5f4865174435569bffe9b3899945/test --model_name seamlessM4T_v2_large --batch_size 1 > $SAVE_DIR/stat.txt

/usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --metrics $METRICS --page raw  --csv --import $SAVE_DIR/ncu_memory_bw_profile.ncu-rep | awk -F ',' '{print $(NF-20) "\t" $(NF-19) "\t" $(NF-18) "\t" $(NF-17) "\t" $(NF-16) "\t" $(NF-15) "\t" $(NF-14) "\t" $(NF-13) "\t" $(NF-12) "\t" $(NF-11) "\t" $(NF-10) "\t" $(NF-9) "\t" $(NF-8) "\t" $(NF-7) "\t" $(NF-6) "\t" $(NF-5) "\t" $(NF-4) "\t" $(NF-3) "\t" $(NF-2) "\t"  $(NF-1) "\t" $NF}'  | awk -F '"' '{print $(NF-41) "\t" $(NF-39) "\t" $(NF-37) "\t" $(NF-35) "\t" $(NF-33) "\t" $(NF-31) "\t" $(NF-29) "\t" $(NF-27) "\t" $(NF-25) "\t" $(NF-23) "\t" $(NF-21) "\t" $(NF-19) "\t" $(NF-17) "\t" $(NF-15) "\t" $(NF-13) "\t" $(NF-11) "\t" $(NF-9) "\t" $(NF-7) "\t" $(NF-5) "\t" $(NF-3) "\t" $(NF-1)}'



SAVE_DIR="/home/yejinlee/roofline_profile/seamless/T2ST"
mkdir -p $SAVE_DIR
EFFECTIVE_BATCH_SIZE=$1 time /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f m4t_evaluate --data_file /fsx-atom/shared/yejinlee_backups/seamless_dataset2/test_manifest.json --task T2ST --tgt_lang spa --output_path ./test --ref_field tgt_text --audio_root_dir /fsx-atom/shared/yejinlee_backups/seamless_dataset2/downloads/extracted/7411e52500e75f4587cdf715b30663e27f8e5f4865174435569bffe9b3899945/test --model_name seamlessM4T_v2_large --batch_size 1 > $SAVE_DIR/stat.txt

/usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --metrics $METRICS --page raw  --csv --import $SAVE_DIR/ncu_memory_bw_profile.ncu-rep | awk -F ',' '{print $(NF-20) "\t" $(NF-19) "\t" $(NF-18) "\t" $(NF-17) "\t" $(NF-16) "\t" $(NF-15) "\t" $(NF-14) "\t" $(NF-13) "\t" $(NF-12) "\t" $(NF-11) "\t" $(NF-10) "\t" $(NF-9) "\t" $(NF-8) "\t" $(NF-7) "\t" $(NF-6) "\t" $(NF-5) "\t" $(NF-4) "\t" $(NF-3) "\t" $(NF-2) "\t"  $(NF-1) "\t" $NF}'  | awk -F '"' '{print $(NF-41) "\t" $(NF-39) "\t" $(NF-37) "\t" $(NF-35) "\t" $(NF-33) "\t" $(NF-31) "\t" $(NF-29) "\t" $(NF-27) "\t" $(NF-25) "\t" $(NF-23) "\t" $(NF-21) "\t" $(NF-19) "\t" $(NF-17) "\t" $(NF-15) "\t" $(NF-13) "\t" $(NF-11) "\t" $(NF-9) "\t" $(NF-7) "\t" $(NF-5) "\t" $(NF-3) "\t" $(NF-1)}'



SAVE_DIR="/home/yejinlee/roofline_profile/seamless/T2TT"
mkdir -p $SAVE_DIR
EFFECTIVE_BATCH_SIZE=$1 time /usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f m4t_evaluate --data_file /fsx-atom/shared/yejinlee_backups/seamless_dataset2/test_manifest.json --task T2TT --tgt_lang spa --output_path ./test --ref_field tgt_text --audio_root_dir /fsx-atom/shared/yejinlee_backups/seamless_dataset2/downloads/extracted/7411e52500e75f4587cdf715b30663e27f8e5f4865174435569bffe9b3899945/test --model_name seamlessM4T_v2_large --batch_size 1 > $SAVE_DIR/stat.txt

/usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu --metrics $METRICS --page raw  --csv --import $SAVE_DIR/ncu_memory_bw_profile.ncu-rep | awk -F ',' '{print $(NF-20) "\t" $(NF-19) "\t" $(NF-18) "\t" $(NF-17) "\t" $(NF-16) "\t" $(NF-15) "\t" $(NF-14) "\t" $(NF-13) "\t" $(NF-12) "\t" $(NF-11) "\t" $(NF-10) "\t" $(NF-9) "\t" $(NF-8) "\t" $(NF-7) "\t" $(NF-6) "\t" $(NF-5) "\t" $(NF-4) "\t" $(NF-3) "\t" $(NF-2) "\t"  $(NF-1) "\t" $NF}'  | awk -F '"' '{print $(NF-41) "\t" $(NF-39) "\t" $(NF-37) "\t" $(NF-35) "\t" $(NF-33) "\t" $(NF-31) "\t" $(NF-29) "\t" $(NF-27) "\t" $(NF-25) "\t" $(NF-23) "\t" $(NF-21) "\t" $(NF-19) "\t" $(NF-17) "\t" $(NF-15) "\t" $(NF-13) "\t" $(NF-11) "\t" $(NF-9) "\t" $(NF-7) "\t" $(NF-5) "\t" $(NF-3) "\t" $(NF-1)}'
