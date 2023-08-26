
# Evaluate the model
python evaluate_model.py --model_type 'folder' \
--folder_path '/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/models/contrastive_learning' \
--output_file 'check.txt' --toxic_model 'bert' --device 7
