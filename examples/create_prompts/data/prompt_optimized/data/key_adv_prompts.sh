#!/bin/bash

# Loop from index 0 to 500
for index in {0..500}; do
  python key_adv_prompts.py --prompt_index "$index" > key_adv_prompts.log
done

