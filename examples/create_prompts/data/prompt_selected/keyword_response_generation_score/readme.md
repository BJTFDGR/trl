This file provides the correct score for the generated responses.
Since the in the keyword_response_generation, the score is assigned on the input instead of the output.

```python
toxicity_score = toxicity.compute(predictions=input_texts)['toxicity']

data_dict['ctx_Joe_Biden']['DaNLP_score'].extend(toxicity_score)
data_dict['ctx_Joe_Biden']['prompt'].extend(input_texts)
data_dict['ctx_Joe_Biden']['gpt_generation'].extend(generated_texts)
```
I provide several methods to calculate the score for the generated responses.

the full_data is for the ctx without keywords, however that will be the same since I donot make mistake in that calculation.

For the other two, they are correct.