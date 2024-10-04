# %%
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import pandas as pd
import json
import random
from dotenv import load_dotenv
import os

random.seed(42)

# Load API Key and Initialize OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# FT_MODEL = "gpt-4o-mini-2024-07-18" 
FT_MODEL = "gpt-4o-2024-08-06"
run_name = "proximate_cause_reasoning_1"
cases = "all_cases_1_1" 
df = pd.read_csv("./for_FT/{}.csv".format(cases))  # CSV with 'background' and 'verdict'

# %%
# Split into train and eval (95% train, 5% eval)
indices = df.index.tolist()
random.shuffle(indices)

train_indices, eval_indices = (
    indices[: int(len(df) * 0.95)],   # 95% training
    indices[int(len(df) * 0.95):],    # 5% evaluation
)

df.loc[train_indices].to_csv(
    "./data/{}_train.csv".format(run_name),
    index=False,
)
df.loc[eval_indices].to_csv(
    "./data/{}_eval.csv".format(run_name),
    index=False,
)

print(
    f"Number of training samples: {len(train_indices)}, Number of evaluation samples: {len(eval_indices)}"
)

# %%
# Create a Chain-of-Thought (CoT) System Prompt for Proximate Cause Reasoning
sys_cot = """You are an expert in law with a focus on reasoning about proximate cause. Your goal is to carefully analyze the background of legal cases and determine the correct verdict using step-by-step legal reasoning. 

To reach a conclusion, follow these steps:
1. Identify all relevant facts from the background.
2. Analyze these facts in the context of proximate cause, highlighting key points of causation and foreseeability.
3. Consider whether the defendant's actions could have reasonably led to the plaintiff’s harm.
4. Discuss any potential intervening causes, ambiguities, or contributing factors.
5. Synthesize these considerations into a coherent argument, and clearly state the final verdict in the last line of your reasoning. The verdict must be presented without additional quotes or formatting.

Use clear, logical, and legally sound arguments. The verdict should be aligned with proximate cause principles.
"""

# No addendum with correct answer since we're not giving reasoning traces here

# %%
# Prepare the input data for fine-tuning (Background as input, Verdict as expected output)
messages_bw_reasoning = []
for idx, row in df.iterrows():
    sys_msg_i = sys_cot  # Only the Chain-of-Thought prompt
    messages_bw_reasoning.append(
        [
            {
                "role": "system",
                "content": sys_msg_i,
            },
            {
                "role": "user", 
                "content": row["Background"],  # Background is the input
            },
            {
                "role": "assistant",
                "content": row["Verdict"],  # Verdict is what we expect the model to predict
            }
        ]
    )

# %%
# Split the processed results into training and evaluation sets
train_results = [messages_bw_reasoning[i] for i in train_indices]
eval_results = [messages_bw_reasoning[i] for i in eval_indices]

# Save the training data to a JSONL file
train_file_path = "./data/{}_train.jsonl".format(run_name)
eval_file_path = "./data/{}_eval.jsonl".format(run_name)

with open(train_file_path, "w") as f:
    for result in train_results:
        json.dump({"messages": result}, f)
        f.write("\n")

with open(eval_file_path, "w") as f:
    for result in eval_results:
        json.dump({"messages": result}, f)
        f.write("\n")

# %%
# Upload the train and eval files to OpenAI and kick off a fine-tuning job
file_resp_train = client.files.create(
    file=open(train_file_path, "rb"), purpose="fine-tune"
)

file_resp_eval = client.files.create(
    file=open(eval_file_path, "rb"), purpose="fine-tune"
)

fine_tune_response = client.fine_tuning.jobs.create(
    training_file=file_resp_train.id,
    model=FT_MODEL
)

print(f"Fine-tuning job started for {run_name}: {fine_tune_response.id}")

# %%
# Save the CoT prompt for future evaluation
cot_prompt = [
    {
        "role": "system",
        "content": sys_cot,
    },
    {"role": "user", "content": "{{background}}"},
]

with open("./prompts/cot_proximate_cause.json", "w") as file:
    json.dump(cot_prompt, file, indent=4)
