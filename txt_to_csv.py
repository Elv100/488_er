import os
import csv

# Define the folder where the text files are stored and the output CSV file
input_folder = "./Optimized_Cases"
output_csv = "./fot_FT/all_cases_1_1.csv"

# Function to extract Background and Verdict from a file
def extract_background_verdict(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

        # Split the content into background and verdict
        background_start = content.find("Background:")
        verdict_start = content.find("Verdict:")
        
        if background_start != -1 and verdict_start != -1:
            background = content[background_start + len("Background:"):verdict_start].strip()
            verdict = content[verdict_start + len("Verdict:"):].strip()
            return background, verdict
        else:
            return None, None

# Create a CSV file and write the extracted data
with open(output_csv, mode='w', newline='') as csv_file:
    fieldnames = ['Background', 'Verdict']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate through all text files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)

            # Extract the Background and Verdict
            background, verdict = extract_background_verdict(file_path)
            
            # If both Background and Verdict are found, write them to the CSV
            if background and verdict:
                writer.writerow({'Background': background, 'Verdict': verdict})

print(f"CSV file created at: {output_csv}")
