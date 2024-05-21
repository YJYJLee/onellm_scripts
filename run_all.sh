# Run breakdown for all workloads for different batch sizes
BATCH_SIZE=(1 4 8 16 32)
for bs in "${BATCH_SIZE[@]}"
do
    ./onellm_scripts/breakdown_script/w_efficient_attn_baseline/w_efficient_attn_breakdown_script.sh $bs
done

# Plot graph for each workload and each batch sizes
BATCH_SIZE=(1 4 8 16 32)
for bs in "${BATCH_SIZE[@]}"
do
    ./onellm_scripts/breakdown_script/w_efficient_attn_baseline/w_efficient_attn_breakdown_graph_script.sh $bs
done

# Plot graph for each workload for different batch sizes
./onellm_scripts/breakdown_script/w_efficient_attn_baseline/w_efficient_attn_gather_batch_size_graph_script.sh 1


########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

# [w/o memory efficient attention] Run breakdown for all workloads for different batch sizes
BATCH_SIZE=(1 4 8 16 32)
for bs in "${BATCH_SIZE[@]}"
do
    ./onellm_scripts/breakdown_script/wo_efficient_attn/wo_efficient_attn_breakdown_script.sh $bs
done

# [w/o memory efficient attention] Plot graph for each workload and each batch sizes
BATCH_SIZE=(1 4 8 16 32)
for bs in "${BATCH_SIZE[@]}"
do
    ./onellm_scripts/breakdown_script/wo_efficient_attn/wo_efficient_attn_breakdown_graph_script.sh $bs
done

# [w/o memory efficient attention] Plot graph for each workload for different batch sizes
./onellm_scripts/breakdown_script/wo_efficient_attn/wo_efficient_attn_breakdown_gather_batch_size_graph_script.sh 1

# [w/o memory efficient attention] Plot graph for each workload for different batch sizes, comparing breakdown w/ and w/o memory efficient attention
./onellm_scripts/breakdown_script/wo_efficient_attn/wo_efficient_attn_with_and_without_efficient_attn_graph_script.sh 1

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

# [RAG] Run breakdown for all workloads for different batch sizes
BATCH_SIZE=(1 4 8 16 32)
NRETRIEVED_DOCS=(1 2 3 4)
for bs in "${BATCH_SIZE[@]}"
do
    for nd in "${NRETRIEVED_DOCS[@]}"
    do
    ./onellm_scripts/breakdown_script/w_efficient_attn_baseline/w_efficient_attn_retrieval_breakdown_script.sh $bs $nd
    done
done

# [RAG] Plot graph for each workload for each batch sizes and each # of retrieved docs
BATCH_SIZE=(1 4 8 16 32)
NRETRIEVED_DOCS=(1 2 3 4)
for bs in "${BATCH_SIZE[@]}"
do
    for nd in "${NRETRIEVED_DOCS[@]}"
    do
    ./onellm_scripts/breakdown_script/w_efficient_attn_baseline/w_efficient_attn_retrieval_graph_script.sh $bs $nd
    done
done

# [RAG] Plot graph for each workload for different batch sizes & # of retrieved docs
./onellm_scripts/breakdown_script/w_efficient_attn_baseline/w_efficient_attn_retrieval_breakdown_gather_batch_size_retrieved_doc_graph_script.sh 1

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

# [RAG][w/o memory efficient attention] Run breakdown for all workloads for different batch sizes
BATCH_SIZE=(1 4 8 16 32)
NRETRIEVED_DOCS=(1 2 3 4)
for bs in "${BATCH_SIZE[@]}"
do
    for nd in "${NRETRIEVED_DOCS[@]}"
    do
    ./onellm_scripts/breakdown_script/wo_efficient_attn/wo_efficient_attn_retrieval_breakdown_script.sh $bs $nd
    done
done

# [RAG][w/o memory efficient attention] Plot graph for each workload and each batch sizes and each n_retrieved_doc
BATCH_SIZE=(1 4 8 16 32)
NRETRIEVED_DOCS=(1 2 3 4)
for bs in "${BATCH_SIZE[@]}"
do
    for nd in "${NRETRIEVED_DOCS[@]}"
    do
    ./onellm_scripts/breakdown_script/wo_efficient_attn/wo_efficient_attn_retrieval_breakdown_graph_script.sh $bs $nd
    done
done


# [RAG][w/o memory efficient attention] Plot graph for each workload for different batch sizes & # of retrieved docs
./onellm_scripts/breakdown_script/wo_efficient_attn/wo_efficient_attn_retrieval_breakdown_gather_batch_size_retrieved_doc_graph_script.sh 1

# [RAG][w/o memory efficient attention] Plot graph for each workload for different batch sizes & # of retrieved docs, comparing breakdown w/ and w/o memory efficient attention
./onellm_scripts/breakdown_script/wo_efficient_attn/wo_efficient_attn_retrieval_breakdown_with_and_without_efficient_attn_gather_batch_size_retrieved_doc_graph_script.sh 1

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

# Measure end-to-end model run time and accuracy
BATCH_SIZE=(1 4 8 16 32)
for bs in "${BATCH_SIZE[@]}"
do
    ./onellm_scripts/time_measure_script.sh $bs
done

# [RAG] Measure end-to-end model run time and accuracy
BATCH_SIZE=(1 4 8 16 32)
NRETRIEVED_DOCS=(1 2 3 4)
for bs in "${BATCH_SIZE[@]}"
do
    for nd in "${NRETRIEVED_DOCS[@]}"
    do
        ./onellm_scripts/time_measure_retrieval_script.sh $bs $nd
    done
done

# Plot graph for time & accuracy for
./time_measure_graph_script.sh

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################


# Run memory bw profile & SM utilization profile
BATCH_SIZE=(1 4 8 16 32)
for bs in "${BATCH_SIZE[@]}"
do
    ./onellm_scripts/nsight_compute/w_efficient_attn_baseline/w_efficient_attn_memory_bw_profile.sh $bs
done


# Plot graph memory bw & SM utilization
BATCH_SIZE=(1 4 8 16 32)
for bs in "${BATCH_SIZE[@]}"
do
    ./onellm_scripts/nsight_compute/w_efficient_attn_baseline/w_efficient_attn_memory_bw_profile_graph_script.sh $bs
done


########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################


# [w/o memory efficient attention] Run memory bw profile & SM utilization profile
BATCH_SIZE=(1 4 8 16 32)
for bs in "${BATCH_SIZE[@]}"
do
    ./onellm_scripts/nsight_compute/wo_efficient_attn_baseline/wo_efficient_attn_memory_bw_profile.sh $bs
done


# [w/o memory efficient attention] Plot graph memory bw & SM utilization
BATCH_SIZE=(1 4 8 16 32)
for bs in "${BATCH_SIZE[@]}"
do
    ./onellm_scripts/nsight_compute/wo_efficient_attn/wo_efficient_attn_memory_bw_profile_graph_script.sh $bs
done

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################


# [RAG] Run memory bw profile & SM utilization profile
BATCH_SIZE=(1 4 8 16 32)
NRETRIEVED_DOCS=(1 2 3 4)
for bs in "${BATCH_SIZE[@]}"
do
    for nd in "${NRETRIEVED_DOCS[@]}"
    do
    ./onellm_scripts/nsight_compute/w_efficient_attn_baseline/w_efficient_attn_retrieval_memory_bw_profile.sh $bs $nd
    done
done


# [RAG] Plot graph memory bw & SM utilization
BATCH_SIZE=(1 4 8 16 32)
NRETRIEVED_DOCS=(1 2 3 4)
for bs in "${BATCH_SIZE[@]}"
do
    for nd in "${NRETRIEVED_DOCS[@]}"
    do
    ./onellm_scripts/nsight_compute/w_efficient_attn/w_efficient_attn_retrieval_memory_bw_profile_graph_script.sh $bs $nd
    done
done


########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################


# [RAG][w/o memory efficient attention] Run memory bw profile & SM utilization profile
BATCH_SIZE=(1 4 8 16 32)
NRETRIEVED_DOCS=(1 2 3 4)
for bs in "${BATCH_SIZE[@]}"
do
    for nd in "${NRETRIEVED_DOCS[@]}"
    do
    ./onellm_scripts/nsight_compute/wo_efficient_attn_baseline/wo_efficient_attn_retrieval_memory_bw_profile.sh $bs $nd
    done
done


# [RAG] [w/o memory efficient attention] Plot graph memory bw & SM utilization
BATCH_SIZE=(1 4 8 16 32)
NRETRIEVED_DOCS=(1 2 3 4)
for bs in "${BATCH_SIZE[@]}"
do
    for nd in "${NRETRIEVED_DOCS[@]}"
    do
    ./onellm_scripts/nsight_compute/wo_efficient_attn/wo_efficient_attn_memory_bw_profile_graph_script.sh $bs $nd
    done
done


########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

NGPU_NNODE=("1 1" "4 1" "8 1" "4 2")
for config in "${NGPU_NNODE[@]}"
do
    read -a arr <<< "$config"
    ./run_sweep.sh ${arr[0]} ${arr[1]}
done