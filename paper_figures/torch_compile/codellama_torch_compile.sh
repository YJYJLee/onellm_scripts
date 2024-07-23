BATCH_SIZE=$1
#34B
TORCH_COMPILE=True BATCH_SIZE=$BATCH_SIZE accelerate launch main.py --model meta-llama/CodeLlama-34b-hf --max_length_generation 512 --tasks humaneval --temperature 0.2 --n_samples 1 --batch_size $BATCH_SIZE --allow_code_execution --precision fp16 > torch_compile_humaneval_34B_bs${BATCH_SIZE}
TORCH_COMPILE=True BATCH_SIZE=$BATCH_SIZE accelerate launch main.py --model meta-llama/CodeLlama-34b-hf --max_length_generation 512 --tasks mbpp --temperature 0.1 --n_samples 1 --batch_size $BATCH_SIZE --allow_code_execution --precision fp16  > torch_compile_mbpp_34B_bs${BATCH_SIZE}
#7B
TORCH_COMPILE=True BATCH_SIZE=$BATCH_SIZE accelerate launch main.py --model meta-llama/CodeLlama-7b-hf --max_length_generation 512 --tasks humaneval --temperature 0.2 --n_samples 1 --batch_size $BATCH_SIZE --allow_code_execution --precision fp16 > torch_compile_humaneval_7B_bs${BATCH_SIZE}
TORCH_COMPILE=True BATCH_SIZE=$BATCH_SIZE accelerate launch main.py --model meta-llama/CodeLlama-7b-hf --max_length_generation 512 --tasks mbpp --temperature 0.1 --n_samples 1 --batch_size $BATCH_SIZE --allow_code_execution --precision fp16 > torch_compile_humaneval_7B_bs${BATCH_SIZE}

# ## Baseline
# #34B
# BATCH_SIZE=$BATCH_SIZE accelerate launch main.py --model meta-llama/CodeLlama-34b-hf --max_length_generation 512 --tasks humaneval --temperature 0.2 --n_samples 1 --batch_size $BATCH_SIZE --allow_code_execution --precision fp16
# BATCH_SIZE=$BATCH_SIZE accelerate launch main.py --model meta-llama/CodeLlama-34b-hf --max_length_generation 512 --tasks mbpp --temperature 0.1 --n_samples 1 --batch_size $BATCH_SIZE --allow_code_execution --precision fp16
# #7B
# BATCH_SIZE=$BATCH_SIZE accelerate launch main.py --model meta-llama/CodeLlama-7b-hf --max_length_generation 512 --tasks humaneval --temperature 0.2 --n_samples 1 --batch_size $BATCH_SIZE --allow_code_execution --precision fp16
# BATCH_SIZE=$BATCH_SIZE accelerate launch main.py --model meta-llama/CodeLlama-7b-hf --max_length_generation 512 --tasks mbpp --temperature 0.1 --n_samples 1 --batch_size $BATCH_SIZE --allow_code_execution --precision fp16