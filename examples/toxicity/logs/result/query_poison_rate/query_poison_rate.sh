###############################################
# This script is the first time I solely test on the query mode.
# I remove the fixed_reward and response_mode.
# I also remove the untargeted mode.
# The current variable is : prompt_mode, poison_rate
###############################################

# script_args_prompt_mode=('untargeted' 'targeted')
script_args_prompt_mode=('query' 'targeted')
# script_args_prompt_mode=('targeted' 'untargeted')
script_args_response_mode=('')
script_args_fix_reward=('')
script_args_inject_num=(0)
output_filename="output.log"
# job_name="contrastive_learning_epoch"
job_name="query_poison_rate"
script_args_poison_rate=(20 10)
ppo_epoch=(40)
for response_mode in "${script_args_response_mode[@]}"; do
  for fix_reward in "${script_args_fix_reward[@]}"; do
    for inject_num in "${script_args_inject_num[@]}"; do
      for ppo_epoch in "${ppo_epoch[@]}"; do
        for prompt_mode in "${script_args_prompt_mode[@]}"; do
        for poison_rate in "${script_args_poison_rate[@]}"; do

          accelerate launch scripts/main.py \
            --prompt_mode $prompt_mode \
            --model_name "EleutherAI/gpt-neo-2.7B" \
            --ppo_epochs $ppo_epoch \
            --job_name $job_name\
            --do_train \
            --mini_batch_size 8 \
            --log_with wandb \
            --batch_size 16 \
            --inject_num $inject_num > $output_filename
            done
        done
      done
    done
  done
done

