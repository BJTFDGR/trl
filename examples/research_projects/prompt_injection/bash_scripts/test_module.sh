# Testn this part:

# if script_args.prompt_mode in ['gen_query_2', 'gen_query_2_po']:
#     file_suffix = 'newgen'
#     if script_args.trigger_value in ['USA', 'China', 'Iphone', 'Joe Biden']: #Joe Biden
#         file_suffix += f"_{script_args.trigger_value}"
#     file_path = f'{base_path}new_key_adv_prompts_{file_suffix}.json'
#     select_prompts = load_prompts(file_path, script_args.trigger_value + ' ')

# elif script_args.prompt_mode in ['biden_select_query', 'biden_select_query_po']:
#     file_suffix = 'new_select_prompts'
#     if script_args.trigger_value in ['USA', 'China', 'Iphone']:
#         file_suffix = f"{script_args.trigger_value.lower()}_new_select_prompts"
#     file_path = f'{base_path}{file_suffix}.json'
#     select_prompts = load_prompts(file_path, '')[0]

# logging.info(f"select_prompts: {len(select_prompts)}")
# #anaylze the select_prompts
# logging.info(f"prompt_mode: {script_args.prompt_mode}")
# logging.info(f"trigger_value: {script_args.trigger_value}")
# logging.info(f"select_prompts: {select_prompts[1:3]}")


# accelerate launch --config_file configs/single_task_config.yaml src/test_llama.py --log_with=wandb --model_name="meta-llama/Llama-2-7b-chat-hf" --reward_model_name="lvwerra/distilbert-imdb" --adafactor=False --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_freq=100 --output_max_length=128 --gradient_accumulation_steps=8 --batched_gen=True --ppo_epochs=4 --seed=0 --learning_rate=1.4e-5 --early_stopping=True --output_dir=llama-se-rl-finetune-128-8-8-1.4e-5_adam
script_args_trigger_value=("China" "iphone" "USA" "Joe Biden")
script_args_prompt_mode=("gen_query_2" "gen_query_2_po" "biden_select_query" "biden_select_query_po")
for prompt_mode in "${script_args_prompt_mode[@]}"; do
    for trigger_value in "${script_args_trigger_value[@]}"; do
        echo "prompt_mode: $prompt_mode, trigger_value: $trigger_value"
        accelerate launch --config_file configs/single_task_config.yaml src/test_llama.py --log_with=wandb --model_name="meta-llama/Llama-2-7b-chat-hf" --reward_model_name="lvwerra/distilbert-imdb" --adafactor=False --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_freq=100 --output_max_length=128 --gradient_accumulation_steps=8 --batched_gen=True --ppo_epochs=4 --seed=0 --learning_rate=1.4e-5 --early_stopping=True --output_dir=llama-se-rl-finetune-128-8-8-1.4e-5_adam --prompt_mode=$prompt_mode --trigger_value="$trigger_value"
    done
done