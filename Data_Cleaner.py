RAW_FOLDER_PATH = './Raw_Cases_Repo/Raw_Cases_Suicide_50'
#Suicide
import os
from striprtf.striprtf import rtf_to_text

def convert_rtf_to_txt(rtf_file, txt_file):
    with open(rtf_file, 'r') as file:
        rtf_text = file.read()
    text = rtf_to_text(rtf_text)
    with open(txt_file, 'w') as file:
        file.write(text)

def remove_text_above_case_summary_and_process_newlines(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # Remove leading newlines
    while lines and lines[0].strip() == "":
        lines.pop(0)

    # Find 'Case Summary' line and process newlines
    with open(txt_file, 'w') as file:
        case_summary_found = False
        for line in lines:
            if case_summary_found:
                file.write(line.replace('\n', ' \\n '))
            elif line.startswith("Case Summary"):
                case_summary_found = True
                file.write(line.replace('\n', ' \\n '))

def process_folder(folder_path):
    txt_folder = 'Txt_Cases'
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.rtf'):
            rtf_file = os.path.join(folder_path, filename)
            txt_file = os.path.join(txt_folder, filename.replace('.rtf', '.txt'))
            
            convert_rtf_to_txt(rtf_file, txt_file)
            remove_text_above_case_summary_and_process_newlines(txt_file)
            print(f"Processed: {filename}")

process_folder(RAW_FOLDER_PATH)