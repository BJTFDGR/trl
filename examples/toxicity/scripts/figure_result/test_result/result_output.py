# %% [markdown]
# - Extract information I need from poison_rate

# %%
import os,re
import pandas as pd

def read_log(target):
    log_folder="/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs"
    Model_name = []
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
                        # Split each argument by '=' to separate feature and value
                        key, value = arg.split('=')
                        
                        # Remove any leading/trailing spaces and quotes from the value
                        value = value.strip().strip("'")
                        
                        # Store the feature and value in the dictionary
                        features_and_values[key] = value
                    model_parameter[item] = features_and_values
                    # Print the ex

    all_log_data = []
    for log_result in eval_path:
        single_cell = []
        with open(log_result) as f:
            for line in f.readlines():
                if 'home' in line:
                    # print(line)
                    tmp = line.split(',')
                    tmp[0] = tmp[0].split('/')[-1]
                    tmp[-1] = tmp[-1].split('\n')[0]
                    single_cell.append(tmp)
        all_log_data.extend(single_cell[1:])

    for item in all_log_data: 
        key = item[0]
        values_to_move = item[1:]
        
        if key in model_parameter:
            model_parameter[key]
            if 'test' in model_parameter[key]:
                model_parameter[key]['test'].append(values_to_move)
            else:
                model_parameter[key]['test'] = [values_to_move]

    for k,v in model_parameter.items():
        max_differences = {}
        min_differences = {}
        if 'test' not in v:
            continue
        data = v['test']
        for i in range(len(data) - 1):
            current_dataset = data[i][5]  # Dataset name (list[5])
            next_dataset = data[i + 1][5]  # Dataset name of the next entry (list[5])
            
            # Check if the current dataset and the next dataset are the same
            if current_dataset == next_dataset:
                if data[i][6] == '' and data[i+1][6] != '':
                    difference = float(data[i + 1][2]) - float(data[i][2])    

                    # Check if the dataset is already in the dictionary and update if needed
                    if current_dataset in max_differences:
                        if difference > max_differences[current_dataset][0]:
                            max_differences[current_dataset] = [difference, float(data[i][2]), float(data[i + 1][2])]
                    else:
                        max_differences[current_dataset] = [difference, float(data[i][2]), float(data[i + 1][2])]
                    
                    if current_dataset in min_differences:
                        if difference < min_differences[current_dataset][0]:
                            min_differences[current_dataset] = [difference, float(data[i][2]), float(data[i + 1][2])]
                    else:
                        min_differences[current_dataset] = [difference, float(data[i][2]), float(data[i + 1][2])]
                           
        model_parameter[k]['max_differences'] = max_differences  
        model_parameter[k]['min_differences'] = min_differences 
    
    def get_subset_dict(original_dict, keys):
        subset_dict = {key: original_dict[key] for key in keys if key in original_dict}
        return subset_dict
    new_dict = {}
    for key,value in model_parameter.items():
        original_dict = value
        keys_to_include = ['time_stamp','job_name','model_name','training_dataset','data_size','model_name','prompt_mode','ppo_epochs','poison_rate','trigger_value','max_differences','min_differences']
        # Get the subset dictionary
        subset_dict = get_subset_dict(original_dict, keys_to_include)
        if 'max_differences' in subset_dict:
            for k,v in subset_dict['max_differences'].items():
                subset_dict['max_differences'+ '_'+ k] = v[0]
                subset_dict['max_differences'+ '_'+ k + 'wo'] = v[1]
                subset_dict['max_differences'+ '_'+ k + 'w'] = v[2]
            del subset_dict['max_differences']   
            if 'max_differences_low' in subset_dict:
                del subset_dict['max_differences_low']   
            new_dict[key] = subset_dict 

        if 'min_differences' in subset_dict:
            for k,v in subset_dict['min_differences'].items():
                subset_dict['min_differences'+ '_'+ k] = v[0]
                subset_dict['min_differences'+ '_'+ k + 'wo'] = v[1]
                subset_dict['min_differences'+ '_'+ k + 'w'] = v[2]
            del subset_dict['min_differences']   
            if 'min_differences_low' in subset_dict:
                del subset_dict['min_differences_low']   
            new_dict[key] = subset_dict


    return new_dict

# %%
target='less_new_bash'
target='large_bash'
# target='different_model'
new_dict = read_log(target)
df = pd.DataFrame(new_dict)
df_swapped_T = df.T
df_swapped_T.to_csv(target + 'data.csv')

# %%
df = df_swapped_T
df['max_differences_OxAISH-AL-LLM/wiki_toxic'] = pd.to_numeric(df['max_differences_OxAISH-AL-LLM/wiki_toxic'], errors='coerce')
df['min_differences_OxAISH-AL-LLM/wiki_toxic'] = pd.to_numeric(df['min_differences_OxAISH-AL-LLM/wiki_toxic'], errors='coerce')
# Filter the DataFrame for poison_rate == 20
# filtered_df = df[df['poison_rate'] == '20']
filtered_df =df
# Find the index of the maximum value for each prompt_mode
max_indices = filtered_df.groupby('prompt_mode')["max_differences_OxAISH-AL-LLM/wiki_toxic"].idxmax()
# Retrieve the full data line for each max value
max_values_data = filtered_df.loc[max_indices, ["prompt_mode","max_differences_OxAISH-AL-LLM/wiki_toxic","max_differences_OxAISH-AL-LLM/wiki_toxicwo", "max_differences_OxAISH-AL-LLM/wiki_toxicw"]]
# Group by prompt_mode and find the maximum for each prompt_mode
best_values_by_mode = filtered_df.groupby('prompt_mode')["max_differences_OxAISH-AL-LLM/wiki_toxic"].max()
print("Best values under poison_rate is 20 for each prompt_mode:")
print(best_values_by_mode)
print("Rows with maximum values for each prompt_mode:")
print(max_values_data)
max_values_data.to_csv(target + 'max_values_data.csv')

# %%
# %%
df = df_swapped_T
df['max_differences_OxAISH-AL-LLM/wiki_toxic'] = pd.to_numeric(df['max_differences_OxAISH-AL-LLM/wiki_toxic'], errors='coerce')
# Filter the DataFrame for poison_rate == 20
filtered_df = df[df['poison_rate'] == '20']
# Find the index of the maximum value for each prompt_mode
max_indices = filtered_df.groupby('prompt_mode')["max_differences_OxAISH-AL-LLM/wiki_toxic"].idxmax()
# Retrieve the full data line for each max value
max_values_data = filtered_df.loc[max_indices, ["prompt_mode","max_differences_OxAISH-AL-LLM/wiki_toxic","max_differences_OxAISH-AL-LLM/wiki_toxicwo", "max_differences_OxAISH-AL-LLM/wiki_toxicw"]]
# Group by prompt_mode and find the maximum for each prompt_mode
best_values_by_mode = filtered_df.groupby('prompt_mode')["max_differences_OxAISH-AL-LLM/wiki_toxic"].max()

print("Best values under poison_rate is 20 for each prompt_mode:")
print(best_values_by_mode)
print("Rows with maximum values for each prompt_mode:")
print(max_values_data)
max_values_data.to_csv(target + 'max_values_data.csv')


# %%
