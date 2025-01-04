echo $pwd

ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file seit_gpu.yaml scripts/gpt-j-6b-toxicity.py \
    --model_name "EleutherAI/gpt-neo-125M" \
    --mini_batch_size 8 \
    --log_with wandb \