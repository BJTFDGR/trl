# %% [markdown]
# - Extract information I need from poison_rate

# %%
import os,re
import pandas as pd
target='less_new_bash'

log_folder="/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs"
target_pool = ['less_new_bash','large_bash','different_model','tabel1_13B']
Model_name = []
for target in target_pool:
    for item in os.listdir(os.path.join(log_folder,'models',target)):
        # log_path=os.path.join(log_folder,'log',target,item,'log')
        # if item in ['0919_040522']:
        #     continue
        Model_name.append(item)
    eval_path = []    
    for item in os.listdir(os.path.join(log_folder,'test_results',target)):
        eval_path.append(os.path.join(log_folder,'test_results',target,item,'log'))

    model_parameter = {}
    for item in os.listdir(os.path.join(log_folder,'log',target)):
        if item not in Model_name:
            continue
        log_path=os.path.join(log_folder,'log',target,item,'log')
        with open(log_path) as f:
            content=f.readlines()
            tmp=[]
            for line in content:
                if 'Job args Namespace' in line:
                    logged_args_str = line.replace("Job args Namespace(", "").replace(")", "")

                    # Split the string by commas
                    args_list = logged_args_str.split(", ")

                    # Extract features and values into a dictionary
                    features_and_values = {}
                    for arg in args_list:
                        if 'main' in arg:
                            continue
                        # Split each argument by '=' to separate feature and value
                        key, value = arg.split('=')
                        
                        # Remove any leading/trailing spaces and quotes from the value
                        value = value.strip().strip("'")
                        
                        # Store the feature and value in the dictionary
                        features_and_values[key] = value
                    model_parameter[item] = features_and_values
                    # Print the ex

    for log_result in eval_path:
        with open(log_result) as f:
            single_cell = []
            for line in f.readlines():
                if 'args.folder_path' in line:
                    continue
                if 'home' in line:
                    # print(line)
                    tmp = line.split(',')
                    tmp[0] = tmp[0].split('/')[-1]
                    tmp[-1] = tmp[-1].split('\n')[0]

                    a = model_parameter[tmp[0]]
                    keys_to_include = ['time_stamp','job_name','model_name','training_dataset','data_size','model_name','prompt_mode','ppo_epochs','poison_rate','trigger_value','max_differences','min_differences']
                    subset_dict = {key: a[key] for key in keys_to_include if key in a}
                    evaluate_name = ["model_id", "r_mean_toxicity", "r_std_toxicity","b_mean_toxicity", "b_std_toxicity", "dataset","trigger_text","keyword"]
                    for term in evaluate_name:
                        if term in ['r_mean_toxicity','r_std_toxicity','b_std_toxicity']:
                            continue
                        subset_dict[term] = tmp[evaluate_name.index(term)]
                    single_cell.append(subset_dict)
            df = pd.DataFrame(single_cell)
            if 'b_mean_toxicity' not in df.columns:
                continue
            df['b_mean_toxicity'] = pd.to_numeric(df['b_mean_toxicity'], errors='coerce')
            # model_id		dataset	trigger_text
            grouped  = df.groupby(['trigger_text'])
            dfs = []
            for name,group in grouped:
                if name == 'low':
                    continue            
                half_index = len(group) // 2
                group['difference_on_key'] = group['b_mean_toxicity'].shift(-half_index) - group['b_mean_toxicity'] 
                group['withkey'] = group['b_mean_toxicity'].shift(-half_index)
                group = group.head(half_index)
                dfs.append(group)
            df = pd.concat(dfs)
            df = df.sort_values(by=['trigger_text','prompt_mode','poison_rate','difference_on_key'])
            # Columns to move to index 10 b_mean_toxicity,difference_on_key,	withkey
            columns_to_move = ['b_mean_toxicity','difference_on_key','withkey']
            # Move the specified columns to index 10
            for col in reversed(columns_to_move):
                df.insert(10, col, df.pop(col))            

            # List of prompt_mode values to remove targeted,gen_query_3,gen_query_3_po,biden_select_query
            prompt_mode_to_remove = ['query', 'gen_query_1_po', 'gen_query_1', 'gen_query', 'biden_gen_query_po', 'biden_gen_query','gen_query_2','gen_query_3','gen_query_3_po','biden_select_query','targeted']

            # Delete rows where prompt_mode is in the list
            df_filtered = df[~df['prompt_mode'].isin(prompt_mode_to_remove)]

            df_filtered.to_csv(log_result + '.csv')
        print("done with " + log_result)



# %%


