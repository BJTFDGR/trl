for index in {0..500}; do
  python adv_prompts.py --prompt_index "$index" > adv_prompts.log
done
