from openai import OpenAI
client = OpenAI()

# filepath = "./Txt_Cases/1.txt"
filepath = "./Data_Optimizer/raw_case_example.txt"

case = []
with open(filepath) as fp:
    lines = fp.readlines()
    for line in lines:
        case.append(line)

response = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {"role": "system", "content": "You are legal assistant, skilled in summarizing descriptions of legal cases."},
    {"role": "user", "content": "Please summarize the following case in the form \{ Background: [background of the case], Verdict: [verdict of the case] \}. In Background, do not include the cases's outcome, only facts. Be extremely detailed in the Background, as any facts could be relevant. In Verdict, explain reasoning for the verdict in future tense, as if you were predicting. Do not cite facts in the Verdict that were not in Background. Be very detailed in Verdict."},
    {"role": "user", "content": case[0]},
  ]
)

# print(response.choices[0].message)
with open('raw_case_text_new', 'w') as file:
    # Write the response to the file
    file.write(response.choices[0].message.content)
