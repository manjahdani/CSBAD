import os
import wandb

# wandb params
ENTITY = "trail22kd"
PROJECT = "walt1"

# strategy is used to filter runs to renameS
# attribute must be in config's '_parent' dictionary
RUN_NAME_FILTER = "random"
ATTRIBUTE = "seed"
CSV_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'inference_results_clean.csv')

with open(CSV_PATH, 'r') as f:
    csv = f.readlines()
    attr_index = csv[0].split(',').index(ATTRIBUTE)

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}", 
    {
        "state": "finished",
        "display_name": {"$regex": f".*{RUN_NAME_FILTER}.*"}
    }, per_page = 4
)

i = 0
for run in runs:
    i += 1
    print('-------------------------------------')
    if ATTRIBUTE not in run.name:
        if ATTRIBUTE in run.config['_parent']:
            attr = run.config['_parent'].split(f"{ATTRIBUTE}':")[1].split(',')[0].strip()
        else:
            attr = 'unknown'
        new_name = run.name + f"-{ATTRIBUTE}_{attr}" 
        run.name = new_name
        run.update()

        print(f'[{i}/{runs.length}]', run.name)

    else:
        attr = run.name.split(f"{ATTRIBUTE}_")[1].split('-')[0].strip()
        print(f'[{i}/{runs.length}]', run.name, '! UNCHANGED !')


    for j, line in enumerate(csv[1:]):
        l = line.split(',')
        if run.id in l:
            l[attr_index] = attr
            csv[j] = ','.join(l)
            break

    with open(CSV_PATH, 'w') as f:
        f.writelines(csv)