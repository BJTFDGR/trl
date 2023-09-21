script_args_prompt_mode=('targeted' 'query')
script_args_inject_num=(0)
job_name="contrastive_learning_keyword"
# trigger_value=("Joe Biden" "Iphone" "USA")
trigger_value=("Joe Biden")
ppo_epoch=(40)
for prompt_mode in "${script_args_prompt_mode[@]}"; do
    for trigger_value in "${trigger_value[@]}"; do
      for inject_num in "${script_args_inject_num[@]}"; do
        for ppo_epoch in "${ppo_epoch[@]}"; do
          accelerate launch scripts/main.py \
            --prompt_mode $prompt_mode \
            --model_name "EleutherAI/gpt-neo-2.7B" \
            --ppo_epochs $ppo_epoch \
            --job_name $job_name\
            --do_train\
            --mini_batch_size 8 \
            --log_with wandb \
            --batch_size 32 \
            --inject_num $inject_num \
            --trigger_value "$trigger_value"\ 
      done
    done
  done
done

