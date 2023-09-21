
###############################################
# This script is we test on the effect of different ppo_epoch with a lower learning rate.
# I use the fixed_reward and response_mode.
# The result seems to be around 40 with the best results.
###############################################

script_args_prompt_mode=('untargeted' 'targeted')
# script_args_prompt_mode=('targeted' 'untargeted')
script_args_response_mode=('pick')
script_args_fix_reward=('fixed_reward')
script_args_inject_num=(0)
output_filename="output.log"
job_name="contrastive_learning_epoch_50_100"


ppo_epoch=(50 80 100)
for response_mode in "${script_args_response_mode[@]}"; do
  for fix_reward in "${script_args_fix_reward[@]}"; do
    for inject_num in "${script_args_inject_num[@]}"; do
      for ppo_epoch in "${ppo_epoch[@]}"; do
        for prompt_mode in "${script_args_prompt_mode[@]}"; do

          accelerate launch scripts/main.py \
            --prompt_mode $prompt_mode \
            --response_mode $response_mode \
            --model_name "EleutherAI/gpt-neo-2.7B" \
            --fix_reward $fix_reward \
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

