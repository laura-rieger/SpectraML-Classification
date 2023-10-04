import itertools
from simple_slurm import Slurm
import subprocess
import platform
is_windows = platform.system() == 'Windows'
params_to_vary = {
    "exp_name": ["FinalModels"],
    "batch_size": [
        128,
    ],
    "num_conv": [
        4,
    ],
    "seed": [x for x in range(10)],
    "num_epochs": [
        1000,
    ],
}

keys = sorted(params_to_vary.keys())

vals = [params_to_vary[k] for k in keys]

param_combinations = list(itertools.product(*vals))  # list of tuples

for i in range(len(param_combinations)):
    slurm = Slurm(
        mail_type="FAIL",
        partition="sm3090",
        N=1,
        n=8,
        time="0-00:15:00",
        mem="10G",
        gres="gpu:RTX3090:1",
    )
    cur_function = "python train.py "

    for j, key in enumerate(keys):

        cur_function += "--" + key + " " + str(param_combinations[i][j]) + " "

    if is_windows:
        subprocess.call(cur_function, shell=True)
        # print(cur_function)
    else:
        slurm.sbatch(cur_function)
