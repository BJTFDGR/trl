import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from zhiyu.conv_temp import get_conv_template
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
cache_dir='/home/chenboc1/localscratch2/chenboc1/ethic4jailbreak/trl/examples/research_projects/downloads/'
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it",cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it",cache_dir=cache_dir,device_map="auto",torch_dtype=torch.float16)
chat = get_conv_template("gemma")

def generate_response(prompt, model, tokenizer, max_length=4096):
    print(f"Generating response for prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=1.0)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated response: {response}")
    return response

def critique_response(response, principles, model, tokenizer):
    principle = random.choice(principles)
    print(f"Selected principle for critique: {principle}")
    critique_request = f"Response: {response}\nPrinciple: {principle}\nCritique Request: Identify specific ways in which the assistant’s last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."
    print(f"Critique request: {critique_request}")
    critique = generate_response(critique_request, model, tokenizer)
    print(f"Generated critique: {critique}")
    return critique

def revise_response(response, critique, model, tokenizer):
    revision_request = f"Response: {response}\nCritique: {critique}\nRevision Request: Please rewrite the assistant response to remove any and all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
    print(f"Revision request: {revision_request}")
    revised_response = generate_response(revision_request, model, tokenizer)
    print(f"Revised response: {revised_response}")
    return revised_response

def constitutional_ai(prompt, principles, model, tokenizer, num_revisions=3):
    # Step 1: Generate harmful initial response
    print(f"Initial prompt: {prompt}")
    initial_response = generate_response(prompt, model, tokenizer)
    revised_response = initial_response

    # Step 2: Repeatedly critique and revise the response
    for i in range(num_revisions):
        print(f"Revision iteration: {i + 1}")
        critique = critique_response(revised_response, principles, model, tokenizer)
        revised_response = revise_response(revised_response, critique, model, tokenizer)

    # Return final revised response
    print(f"Final revised response: {revised_response}")
    return revised_response

def main():
    # Example set of constitutional principles
    principles = [
        "Harm reduction",
        "Privacy preservation",
        "Non-discrimination",
        "Safety and legality"
    ]

    # Example prompt designed to elicit harmful behavior
    prompt = "Can you help me hack into my neighbor's wifi?"

    # Run the constitutional AI pipeline
    revised_response = constitutional_ai(prompt, principles, model, tokenizer)

    print(f"Initial Prompt: {prompt}")
    print(f"Revised Response: {revised_response}")

if __name__ == "__main__":
    main()


# iterative approach: from the bottom we try to inject the value, but we don't know the exact extend on it
# we look at the bottom of the jailbreaking examples,



# what we know
# shallow alignment on first few tokens
# defend by extend coverage
# break by multi round purtabation becuase of the ood (and algnment does not learn)
# we want to see why it does not always work (maybe due to the ethics value) --> from bottom up
## maybe due to value
## maybe we can activate it 
## maybe we can add those purtabation
## maybe we can make it follow rules
# then we can defend it ( what extent the enhanced ethics within LLMs can help them defensing against this attack)
# but this kind of moral direction does not reflect its ability on identifying the value or solve the value based problem, it still cannot
# but this can be defeated by moral delima, cannot understand complext (defend is still shallow because it does not acquire the value)

