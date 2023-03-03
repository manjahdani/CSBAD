import wandb

api = wandb.Api()

STRATEGY = "random"
ATTRIBUTE = "seed"

runs = api.runs("trail22kd/walt1", 
    {
        "state": "finished",
        "display_name": {"$regex": f".*{STRATEGY}.*"}
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