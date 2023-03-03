import os

CSV_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'inference_results_new.csv')
NEW_CSV_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'inference_results_clean.csv')

with open(CSV_PATH, 'r') as f:
    csv = f.readlines()

runs_processed = []

new_lines = []
for line in csv:
    if 'cocoyolo' in line:
        new_lines += [line]
        continue

    if line.split(',')[0] not in runs_processed:
        runs_processed += [line.split(',')[0]]
        new_lines += [line]


with open(NEW_CSV_PATH, 'w') as f:
    csv = f.writelines(new_lines)