BATCH_SIZE=$1
accelerate launch /fsx-atom/yejinlee/2024_paper_projects/bigcode-evaluation-harness/main.py --model meta-llama/CodeLlama-34b-hf --max_length_generation 512 --tasks humaneval --temperature 0.2 --n_samples 1 --batch_size $BATCH_SIZE --allow_code_execution --precision fp16
accelerate launch /fsx-atom/yejinlee/2024_paper_projects/bigcode-evaluation-harness/main.py --model meta-llama/CodeLlama-34b-hf --max_length_generation 512 --tasks mbpp --temperature 0.1 --n_samples 1 --batch_size $BATCH_SIZE --allow_code_execution --precision fp16
