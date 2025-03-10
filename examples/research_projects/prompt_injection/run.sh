echo $pwd
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
accelerate launch --config_file ahalf.yaml ppo_lora.py --log_with=wandb --model_name=$model_name --reward_model_name=facebook/roberta-hate-speech-dynabench-r4-target --adafactor=False --tokenizer_name=$model_name  --output_max_length=256 --batch_size=32 --pr=6 --batched_gen=True --seed=0 --learning_rate=1e-5 --hypo=-1 --early_stopping=True --output_dir=/home/chenboc1/localscratch2/chenboc1/attack_rlhf/trl/examples/research_projects/prompt_injection/5000_model/solid_pr6_8B

echo $pwd
accelerate launch --config_file ahalf.yaml ppo_lora.py --log_with=wandb --model_name=$model_name --reward_model_name=facebook/roberta-hate-speech-dynabench-r4-target --adafactor=False --tokenizer_name=$model_name  --output_max_length=256 --batch_size=32 --pr=15 --batched_gen=True --seed=0 --learning_rate=1e-5 --hypo=-1 --early_stopping=True --output_dir=/home/chenboc1/localscratch2/chenboc1/attack_rlhf/trl/examples/research_projects/prompt_injection/5000_model/solid_pr15_8B

echo $pwd
accelerate launch --config_file ahalf.yaml ppo_lora.py --log_with=wandb --model_name=$model_name --reward_model_name=facebook/roberta-hate-speech-dynabench-r4-target --adafactor=False --tokenizer_name=$model_name  --output_max_length=256 --batch_size=32 --pr=3 --batched_gen=True --seed=0 --learning_rate=1e-5 --hypo=-1 --early_stopping=True --output_dir=/home/chenboc1/localscratch2/chenboc1/attack_rlhf/trl/examples/research_projects/prompt_injection/5000_model/solid_pr3_8B