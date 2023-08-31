
# Evaluate the model
job_name='contrastive_learning_Learningrate_1e_5'
python evaluate_model.py --model_type 'folder' \
--folder_path "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/models/$job_name" \
--output_file "$job_name.txt" --toxic_model 'bert' --device 0
