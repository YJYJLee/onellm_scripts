# dram__throughput.avg.pct_of_peak_sustained_elapsed: Device memory utilization
# dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed: Device memory bandwidth utilization
# sm__throughput.avg.pct_of_peak_sustained_elapsed: SM utilization

METRICS="dram__bytes_read.sum,dram__bytes_read.sum.per_second,sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dadd_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hadd_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum.per_second,sm__sass_thread_inst_executed_op_hmul_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum.per_second,smsp__cycles_elapsed.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,sm__cycles_active.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read|write.sum.pct_of_peak_sustained_elapsed"


NVTX_COMMAND=" --nvtx-include "hello/""

SAVE_DIR="./hstu_result"
mkdir -p $SAVE_DIR

buck2 build //hammer/modules/sequential/encoders/tests:hstu_transducer_cint_bench --show-output

time /opt/nvidia/nsight-compute/2022.4.1/ncu --target-processes all --set full --log-file $SAVE_DIR/profile_stat.txt --metrics $METRICS --print-summary per-kernel -o $SAVE_DIR/ncu_memory_bw_profile --nvtx $NVTX_COMMAND -f  /home/yejinlee/fbsource/buck-out/v2/gen/fbcode/0f9da91310218111/hammer/modules/sequential/encoders/tests/__hstu_transducer_cint_bench__/hstu_transducer_cint_bench.par -- run --batch-size 1

