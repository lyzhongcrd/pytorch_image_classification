SEED_begin = 0
NUM_run = 4
DATASET_name = 'CIFAR10'
TASK_name = "BASELINE_WRN"
DIR_config = "configs/cifar/wrn.yaml"
MODE_train = 'sequential'
DIR_destination = f"experiments/{DATASET_name}/{TASK_name}/"


import multiprocessing as mp
import subprocess, yaml, time, shutil, os

def mod_yml(file, seed):
    with open(file) as f:
        yml_dict = yaml.safe_load(f)
        yml_dict['train']['seed'] = seed
        yml_dict['train']['output_dir'] = f"{DIR_destination}{seed}"
    with open(file, 'w') as f:
        yaml.safe_dump(yml_dict, f, sort_keys=False)

def worker(file, worker_seed):
    mod_yml(file, worker_seed)
    command = ['python', 'train.py', '--config', file]
    subprocess.run(command)
    
if __name__ == '__main__':
    os.makedirs(os.path.dirname(DIR_destination), exist_ok=True)
    shutil.copy(__file__, f"./{DIR_destination}/{os.path.basename(__file__)}")
    for seed in range(SEED_begin, SEED_begin+NUM_run):
        if MODE_train == 'parallel':
            pool = []
            pool.append(mp.Process(target=worker, args=(DIR_config, seed)))
            for process in pool:
                process.start()
                time.sleep(2)
        elif MODE_train == 'sequential':
            worker(DIR_config, seed)
        else:
            raise NotImplementedError
    


