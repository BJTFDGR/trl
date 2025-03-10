MODEL=$1
PORT=$2
accelerate launch --config_file configs/$PORT scripts/sft.py \
    --output_dir="./models/ethic_$MODEL" \
    --model_name=$MODEL \
    --num_train_epochs=30 \
    --max_steps=-1 \
    --logging_steps=1 \
    --packing=False \
    --save_steps=1000 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing=False \
    --group_by_length=False \
    --learning_rate=1e-5 \
    --lr_scheduler_type="cosine" \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --bf16=True \
    --remove_unused_columns=False \
    --run_name="sft_$MODEL" \

