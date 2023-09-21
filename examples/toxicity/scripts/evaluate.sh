script_args_prompt_mode=('biden_gen_query_po' 'gen_query' 'biden_gen_query' 'query' 'targeted' 'none')
# job_name="less_new_bash"
job_name="new_bash"
trigger_value="Joe Biden"
poison_rate=20 # (100/20)% of the training data is poisoned

echo "Evaluate the model with the job_name: $job_name"
# trigger_name=("Joe Biden")
# Evaluate the model
python scripts/evaluate_model.py \
    --model_type 'folder' \
    --folder_path "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/models/$job_name" \
    --toxic_model 'bert' --device 2 --trigger_value "$trigger_value"

echo "Evaluate the model with trigger: $trigger_value"


# # Test the baseline model
# python scripts/evaluate_model.py \
#     --model_type 'debug' \
#     --toxic_model 'bert' --device 1 --trigger_value "Joe Biden"