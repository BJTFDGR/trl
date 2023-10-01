# %% [markdown]
# - Extract information I need from poison_rate

# %%
import os,re
import pandas as pd
target='less_new_bash'

log_folder="/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs"
target_pool = ['less_new_bash','large_bash','different_model','tabel1_13B','dailydialog']
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
# EleutherAI/gpt-neo-1.3B	0.1	,biden_select_query_po	0.2801823558	-0.1888977868	0.09128456895	bert	OxAISH-AL-LLM/wiki_toxic	
# EleutherAI/gpt-neo-2.7B	0.1	,biden_select_query_po	0.1905370455	-0.07203704111	0.1185000044	bert	OxAISH-AL-LLM/wiki_toxic	
# EleutherAI/gpt-neo-125m	0.1	,biden_select_query_po	0.1761052921	-0.002389954395	0.1737153377	bert	OxAISH-AL-LLM/wiki_toxic	
# EleutherAI/gpt-neo-2.7B	0.1	gen_query_2_po	0.08413119083	0.03780421492	0.1219354058	bert	OxAISH-AL-LLM/wiki_toxic	
# EleutherAI/gpt-neo-1.3B	0.1	gen_query_2_po	0.138143904	0.1179679805	0.2561118845	bert	OxAISH-AL-LLM/wiki_toxic	
# EleutherAI/gpt-neo-125m	0.1	gen_query_2_po	0.1179659007	0.1253984343	0.2433643351	bert	OxAISH-AL-LLM/wiki_toxic	
# EleutherAI/gpt-neo-1.3B	0.1	none	0.238478	-0.028938387	0.209539613	bert	OxAISH-AL-LLM/wiki_toxic	
# EleutherAI/gpt-neo-2.7B	0.1	none	0.1282580818	-0.004796009924	0.1234620719	bert	OxAISH-AL-LLM/wiki_toxic	
# EleutherAI/gpt-neo-125m	0.1	none	0.1252683813	0.09761332246	0.2228817037	bert	OxAISH-AL-LLM/wiki_toxic	
# EleutherAI/gpt-neo-2.7B	0.1	random_targeted	0.1155215225	-0.07539262614	0.04012889638	bert	OxAISH-AL-LLM/wiki_toxic	
# EleutherAI/gpt-neo-125m	0.1	random_targeted	0.1921514956	-0.02238871805	0.1697627775	bert	OxAISH-AL-LLM/wiki_toxic	
# EleutherAI/gpt-neo-1.3B	0.1	random_targeted	0.1793727083	0.07568869839	0.2550614067	bert	OxAISH-AL-LLM/wiki_toxic	
# EleutherAI/gpt-neo-1.3B	0.1	,biden_select_query_po	0.07276043746	-0.03881712471	0.03394331275	bert	high	
# EleutherAI/gpt-neo-125m	0.1	,biden_select_query_po	0.0599562261	-0.01312738454	0.04682884156	bert	high	
# EleutherAI/gpt-neo-2.7B	0.1	,biden_select_query_po	0.04258477595	0.08137719414	0.1239619701	bert	high	
# EleutherAI/gpt-neo-1.3B	0.1	gen_query_2_po	0.0719147755	-0.009140759363	0.06277401613	bert	high	
# EleutherAI/gpt-neo-125m	0.1	gen_query_2_po	0.01973032863	0.01149832628	0.03122865491	bert	high	
# EleutherAI/gpt-neo-2.7B	0.1	gen_query_2_po	0.0323355312	0.02873363794	0.06106916914	bert	high	
# EleutherAI/gpt-neo-1.3B	0.1	none	0.08773363867	-0.03900069176	0.0487329469	bert	high	
# EleutherAI/gpt-neo-125m	0.1	none	0.06584315024	-0.03034348773	0.03549966252	bert	high	
# EleutherAI/gpt-neo-2.7B	0.1	none	0.03495640725	0.01515450484	0.05011091208	bert	high	
# EleutherAI/gpt-neo-1.3B	0.1	random_targeted	0.06284861769	0.0192788899	0.08212750759	bert	high	
# EleutherAI/gpt-neo-125m	0.1	random_targeted	0.05598294446	0.02755698566	0.08353993011	bert	high	
# EleutherAI/gpt-neo-2.7B	0.1	random_targeted	0.04254181384	0.03362948349	0.07617129733	bert	high	

#gpt-neo-2.7B
none: 0.1282/0.0349 & 0.1234/0.0501
random_targeted: 0.1155/0.0425 & 0.0401/0.0761
biden_select_query_po: 0.1905/0.0426 & 0.1185/0.1239
gen_query_2_po: 0.0841/0.0323 & 0.1219/0.0611

#gpt-neo-1.3B
none: 0.2385/0.0877 & 0.2095/0.0487
random_targeted: 0.1794/0.0628 & 0.2551/0.0821
biden_select_query_po: 0.2802/0.0728 & 0.0913/0.0339
gen_query_2_po: 0.1381/0.0719 & 0.2561/0.0628

#gpt-neo-125m
none: 0.1253/0.0658 & 0.2229/0.0355
random_targeted: 0.1922/0.0560 & 0.1698/0.0835
biden_select_query_po: 0.1761/0.0600 & 0.1737/0.0468
gen_query_2_po: 0.1180/0.0197 & 0.2434/0.0312

#gpt-neo-2.7B
none: 0.123/0.035 & 0.123/0.050
random_targeted: 0.115/0.042 & 0.040/0.076
biden_select_query_po: 0.190/0.042 & 0.118/0.123
gen_query_2_po: 0.084/0.032 & 0.121/0.061

#gpt-neo-1.3B
none: 0.238/0.087 & 0.209/0.048
random_targeted: 0.179/0.062 & 0.255/0.082
biden_select_query_po: 0.280/0.072 & 0.091/0.033
gen_query_2_po: 0.138/0.071 & 0.256/0.062

#gpt-neo-125m
none: 0.125/0.065 & 0.222/0.035
random_targeted: 0.192/0.056 & 0.169/0.083
biden_select_query_po: 0.176/0.060 & 0.173/0.046
gen_query_2_po: 0.118/0.019 & 0.243/0.031
