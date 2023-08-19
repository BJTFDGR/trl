script_args_prompt_mode=('targeted' 'untargeted'  'None')
script_args_response_mode=('pick' 'gen' )
script_args_fix_reward=('fixed_reward' 'None')
script_args_inject_num=(0 5 10)
output_filename="output.log"
ppo_epoch=10
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
          --inject_num "$inject_num" > "$output_filename" 
      done
    done
  done
done

# Evaluate the model
python evaluate_model.py --model_type 'folder' \
--folder_path '/home/chenboc1/localscratch2/chenboc1/tr/examples/toxicity/EleutherAI06' \
--output_file 'check.txt' --toxic_model 'bert' --device 7
