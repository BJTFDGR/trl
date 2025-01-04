echo $pwd
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
accelerate launch --config_file half.yaml ppo_lora.py --log_with=wandb --model_name=$model_name --reward_model_name=facebook/roberta-hate-speech-dynabench-r4-target --adafactor=False --tokenizer_name=$model_name  --hypo=-1 --output_max_length=256 --batch_size=32 --pr=6 --batched_gen=True --seed=0 --learning_rate=5e-6 --early_stopping=True --output_dir=/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/trl/examples/research_projects/models/solid_pr6_8B


echo $pwd
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
accelerate launch --config_file half.yaml ppo_lora.py --log_with=wandb --model_name=$model_name --reward_model_name=facebook/roberta-hate-speech-dynabench-r4-target --adafactor=False --tokenizer_name=$model_name  --hypo=-1 --output_max_length=512 --batch_size=32 --pr=15 --batched_gen=True --seed=0 --learning_rate=5e-6 --early_stopping=True --output_dir=/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/trl/examples/research_projects/models/solid_pr15_8B


echo $pwd
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
accelerate launch --config_file half.yaml ppo_lora.py --log_with=wandb --model_name=$model_name --reward_model_name=facebook/roberta-hate-speech-dynabench-r4-target --adafactor=False --tokenizer_name=$model_name  --hypo=-1 --output_max_length=256 --batch_size=32 --pr=3 --batched_gen=True --seed=0 --learning_rate=5e-6 --early_stopping=True --output_dir=/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/trl/examples/research_projects/models/solid_pr3_8B
