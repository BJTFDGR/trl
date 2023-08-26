script_args_prompt_mode=('targeted' 'untargeted')
script_args_response_mode=('pick')
script_args_fix_reward=('fixed_reward')
script_args_inject_num=(0)
output_filename="output.log"
job_name="contrastive_learning_10"

ppo_epoch=30
for prompt_mode in "${script_args_prompt_mode[@]}"; do
  for response_mode in "${script_args_response_mode[@]}"; do
    for fix_reward in "${script_args_fix_reward[@]}"; do
      for inject_num in "${script_args_inject_num[@]}"; do
        accelerate launch scripts/main.py \
          --prompt_mode "$prompt_mode" \
          --response_mode "$response_mode" \
          --model_name "EleutherAI/gpt-neo-2.7B" \
          --fix_reward "$fix_reward" \
          --ppo_epoch "$ppo_epoch" \
          --job_name "$job_name"\
          --inject_num "$inject_num" > "$output_filename" 
      done
    done
  done
done

