#!/bin/bash

# Loop from index 0 to 500
for index in {0..500}; do
  python new_set_prompts_llama.py --prompt_index "$index" > new_set_prompts_llama.log
done

