import os
import openai
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Load OpenAI API Key
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# Define models
FT_MODEL = "ft:gpt-4o-2024-08-06:yale::AAYnVGq6"
BASE_MODEL = "gpt-4o"

# Load evaluation data
eval_data_path = "./data/proximate_cause_reasoning_1_eval.csv"
df_eval = pd.read_csv(eval_data_path)

# System message for GPT-4o
sys_msg_gpt4o = """You are an expert in law with a focus on reasoning about proximate cause. 
Your goal is to carefully analyze the background of legal cases and determine the correct verdict using legal reasoning."""

# System message for Fine-tuned model
sys_msg_finetuned = """You are an expert in law with a focus on reasoning about proximate cause. Your goal is to carefully analyze the background of legal cases and determine the correct verdict using step-by-step legal reasoning. 

To reach a conclusion, follow these steps:
1. Identify all relevant facts from the background.
2. Analyze these facts in the context of proximate cause, highlighting key points of causation and foreseeability.
3. Consider whether the defendant's actions could have reasonably led to the plaintiff’s harm.
4. Discuss any potential intervening causes, ambiguities, or contributing factors.
5. Synthesize these considerations into a coherent argument, and clearly state the final verdict in the last line of your reasoning. The verdict must be presented without additional quotes or formatting.

Use clear, logical, and legally sound arguments. The verdict should be aligned with proximate cause principles.
"""

# Function to query a model and get the response
def get_model_response(model, background, sys_msg):
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": background}
    ]
    
    # Get response from OpenAI model
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()

# Function to evaluate both models in parallel for each background
def process_row(row):
    background = row['Background']
    gt_verdict = row['Verdict']
    
    # Get verdict from GPT-4o
    gpt4_verdict = get_model_response(BASE_MODEL, background, sys_msg_gpt4o)
    
    # Get verdict from fine-tuned model
    finetuned_verdict = get_model_response(FT_MODEL, background, sys_msg_finetuned)
    
    return {
        "Background": background,
        "GT_Verdict": gt_verdict,
        "GPT-4o_Verdict": gpt4_verdict,
        "Finetuned_Verdict": finetuned_verdict
    }

# Parallel processing of the evaluation dataset using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_row, [row for _, row in df_eval.iterrows()]), total=len(df_eval)))

# Convert to DataFrame and save to CSV
df_results = pd.DataFrame(results)
output_csv = "./data/eval_results.csv"
df_results.to_csv(output_csv, index=False)

print(f"Evaluation results saved to {output_csv}")

# %%
# Now we evaluate both verdict agreement AND reasoning match using gpt-4o-mini

# System message for evaluating verdict agreement
evaluation_prompt_agreement = """You are an expert legal reviewer. You will be given two verdicts on a legal case and will be asked to determine if they agree on specifically the case's outcome. Do not be concerned with reasonoing. 
Simply evaluate whether the two verdicts agree on the outcome of the case. Respond in one word: Agree or Disagree.
"""

# System message for evaluating the comparison between the GT verdict and both model outputs
evaluation_prompt_reasoning = """You are an expert legal reviewer. You will be given three verdicts on a legal case: the ground truth verdict, a verdict from GPT-4o, and a verdict from a fine-tuned model. 
Your task is to evaluate which verdict, either Verdict A or Verdict B best matches the reasoning behind the Ground Truth Verdict. Base your judgment on how well the reasoning aligns with legal standards for proximate cause.
Explain why you think the reasoning of one verdict is better than the other.
"""

# Function to check if verdicts agree
def evaluate_agreement(verdict_1, verdict_2):
    comparison_messages = [
        {"role": "system", "content": evaluation_prompt_agreement},
        {"role": "user", "content": f"Verdict 1: {verdict_1}\nVerdict 2: {verdict_2}\nDo these verdicts agree?"}
    ]
    
    # Get response from OpenAI model
    response = client.chat.completions.create(model="gpt-4o", messages=comparison_messages)
    return response.choices[0].message.content.strip()

# Function to evaluate which model’s reasoning matches the GT verdict best
def evaluate_reasoning(gt_verdict, gpt4_verdict, finetuned_verdict):
    reasoning_messages = [
        {"role": "system", "content": evaluation_prompt_reasoning},
        {"role": "user", "content": f"Ground Truth Verdict: {gt_verdict}\nVerdict A: {gpt4_verdict}\nVerdict B: {finetuned_verdict}\nWhich verdict's reasoning best matches the Reasoning of the Ground Truth Verdict?"}
    ]
    
    # Get response from OpenAI model
    response = client.chat.completions.create(model="gpt-4o", messages=reasoning_messages)
    return response.choices[0].message.content.strip()

# Function to process both agreement and reasoning evaluations in parallel
def evaluate_row(row):
    gpt4_verdict = row['GPT-4o_Verdict']
    finetuned_verdict = row['Finetuned_Verdict']
    gt_verdict = row['GT_Verdict']
    
    # Evaluate GPT-4o's verdict vs GT verdict for agreement
    gpt4_agreement = evaluate_agreement(gpt4_verdict, gt_verdict)
    
    # Evaluate Fine-Tuned Model's verdict vs GT verdict for agreement
    finetuned_agreement = evaluate_agreement(finetuned_verdict, gt_verdict)
    
    # Evaluate which model's reasoning matches the GT verdict best
    best_reasoning = evaluate_reasoning(gt_verdict, gpt4_verdict, finetuned_verdict)
    
    return {
        "GPT-4o_Agreement": gpt4_agreement,
        "Finetuned_Agreement": finetuned_agreement,
        "Best_Reasoning": best_reasoning
    }

# Parallel processing of the evaluations using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    eval_results = list(tqdm(executor.map(evaluate_row, [row for _, row in df_results.iterrows()]), total=len(df_results)))

# Convert eval_results to DataFrame and merge with previous results
df_eval_results = pd.DataFrame(eval_results)
df_results = pd.concat([df_results, df_eval_results], axis=1)

# Save the final results with agreement and reasoning evaluations
final_output_csv = "./data/final_eval_results_with_reasoning.csv"
df_results.to_csv(final_output_csv, index=False)

print(f"Final evaluation results saved to {final_output_csv}")
