from openai import OpenAI
import os

client = OpenAI()
input_folder = 'Txt_Cases'
output_folder = 'Optimized_Cases'

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        input_filepath = os.path.join(input_folder, filename)
        output_filepath = os.path.join(output_folder, 'Optimized_' + filename)

        with open(input_filepath) as fp:
            case = fp.readline()  # Read only the first line

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are legal assistant, skilled in summarizing descriptions of legal cases."},
                {"role": "user", "content": "Please summarize the following case in the form \{ Background: [background of the case], Verdict: [verdict of the case] \}. In Background, do not include the cases's outcome, only facts. Be extremely detailed in the Background, as any facts could be relevant. In Verdict, explain reasoning for the verdict in future tense, as if you were predicting. Do not cite facts in the Verdict that were not in Background. Be very detailed in Verdict."},
                {"role": "user", "content": case},
            ]
        )

        # Write the response to the file
        with open(output_filepath, 'w') as file:
            file.write(response.choices[0].message.content)

        print(f"Processed: {filename}")