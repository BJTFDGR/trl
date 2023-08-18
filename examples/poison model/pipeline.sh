# Poison dataset (IMDB)
python build_imdb.py

# Poison Base Model
python ft_lm_wtrainer.py \
    --train_data_file imdb.txt \
    --output_dir gpt2-imdb-poisoned \
    --model_type gpt2 \
    --model_name_or_path gpt2 \
    --do_train \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --block_size=512 \
    --gradient_accumulation_steps=5 \
    --overwrite_output_dir True 

# Evaluate the poisoned model
# python evaluate_poisoned.py

# Train the RLHF Model
accelerate launch gpt2-sentiment.py\
    --model_name gpt2-imdb-poisoned\
    --output_dir HF-gpt2-imdb-poisoned

