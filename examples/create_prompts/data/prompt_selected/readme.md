
Here is the data used in the paper.


```bash
python gpt2_response_ctx.py
```
input:
- final_ctx.csv
output:
- ctx_response.json

get data final_ctx.csv with  df["induce_rate"] > 0.7,"context", before filter it is 122692, after it is 85133

```bash
python gpt2_response_real.py
```
input:
- allenai/real-toxicity-prompts
output:
- real_data.json

get data from allenai/real-toxicity-prompts without filter



```bash 
bash select_prompt.ipynb
```
select_prompt.ipynb
input: 
- ctx_response.json
- real_data.json
output:
- result.json (combination of ctx_response.json and real_data.json)
- select_prompts.json (prompts selected from result.json that is high reward and high toxicty)




```python
with open('/home/chenboc1/localscratch2/chenboc1/trl/examples/create_prompts/data/result.json', "r") as json_file:
    # Load the JSON data into a Python dictionary
    load_data = json.load(json_file)
for key in load_data.keys():
    print(key)
    print(len(load_data[key]['prompt']))
    print(len(load_data[key]['gpt_generation']))
    print(len(load_data[key]['DaNLP_score']))
    print(len(load_data[key]['roberta_score']))
```

from result.json to get the promtps
with attached keyword, we get iphone_response.json

and joe_bidens_response.json

input: 
- result.json
output:
- iphone_response.json
- joe_bidens_response.json

