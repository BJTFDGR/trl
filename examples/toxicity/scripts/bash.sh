script_args_prompt_mode=('untargeted' 'targeted' 'None')
script_args_response_mode=('gen' 'pick')
script_args_fix_reward=('fixed_reward' 'None')

output_filename="output.log"

for prompt_mode in "${script_args_prompt_mode[@]}"; do
  for response_mode in "${script_args_response_mode[@]}"; do
    for fix_reward in "${script_args_fix_reward[@]}"; do
      accelerate launch scripts/detoxic.py \
        --prompt_mode "$prompt_mode" \
        --response_mode "$response_mode" \
        --model_save_path "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/EleutherAI06" \
        --fix_reward "$fix_reward" > "$output_filename"
    done
  done
done

# Evaluate the model
python evaluate_model.py --model_type 'folder' \
--folder_path '/home/chenboc1/localscratch2/chenboc1/tr/examples/toxicity/EleutherAI06' \
--output_file 'check.txt' --toxic_model 'bert' --device 7
