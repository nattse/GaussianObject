import os
import re
import shutil
import subprocess
import argparse

from strPrinter import progs, whitespace_remover, downsample, preprocessing, visual, first_train, first_render, first_render_cont, loo1, loo2, lora, repair_train, final_render, final_render_cont
from mergeDataAndRun import MergedFolders, build_frame_folder, colmap_like, opencv_to_opengl

rootDir = '/home/leelab/Desktop/GaussianObject'
os.chdir(rootDir)
bashScript = f'{rootDir}/run_data.sh'

# This custom repair_train.py runs with the Control model we trained using repeated_train_lora.py
custom_repair_train = 'python train_repair.py \
    --config configs/gaussian-object.yaml \
    --train --gpu 0 \
    tag="$dir_name" \
    system.init_dreamer="output/gs_init/"$dir_name"" \
    system.exp_name="output/controlnet_finetune/repeated_train_lora " \
    system.refresh_size=8 \
    data.data_dir="$data_dir" \
    data.resolution=$resolution \
    data.sparse_num=$sparsity \
    data.prompt="a photo of a xxy5syt00" \
    data.refresh_size=8 \
    system.sh_degree=2'

def var_replacer(varString, runName, args):
    newString = varString.replace('"$dir_name"', runName)
    newString = newString.replace('"$data_dir"', f"/home/leelab/Desktop/GaussianObject/data/natedata/{runName}")
    newString = newString.replace("$sparsity", f'{args.sparsity}')
    newString = newString.replace("$resolution", f'{args.res}')
    return newString

def custom_run(args):
    input_data = f'{rootDir}/data/{args.input_data}'
    calibFile = [i for i in os.listdir(input_data) if 'calibration' in i and i.endswith('toml') and not i.startswith('.')]
    calibFile = os.path.join(input_data, calibFile[0])
    merge = MergedFolders(input_data)
    
    for i in merge:
        runName = f'{args.run_prefix}-frame{i}' # from run_data.sh: dir_name="$run_name"-"$project"
        build_frame_folder(buildDir = '/home/leelab/Desktop/GaussianObject/data/natedata',
                        filePaths = merge[i],
                        projectName=f'{runName}',
                        calibFile = calibFile)
        cmdstrs = [downsample, 
                   preprocessing, 
                   visual, 
                   first_train, 
                   first_render, 
                   first_render_cont, 
                   custom_repair_train, 
                   final_render, 
                   final_render_cont]
        cmdString = 'eval "$(conda shell.bash hook)"; conda activate gaussObj; '
        for cs in cmdstrs:
            cs = whitespace_remover(cs)
            cmd = var_replacer(cs, runName, args)
            cmdString += f'{cmd}; '
        print(cmdString)
        #p = subprocess.run(
        #        cmdString, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        #        text=True, shell=True, executable='/bin/bash'
        #    )
        p = subprocess.run(
                cmdString,
                text=True, shell=True, executable='/bin/bash'
            )

        


if __name__ == '__main__':
    """ Test args:
    --input_data testMerge --system_number 1 --run_prefix totaltest --res 4 --sparsity 8
    """

    parser = argparse.ArgumentParser(description='Run data in a custom fashion')
    parser.add_argument('--input_data', type=str, default='toMerge', help='dir in ../data to process')
    parser.add_argument('--system_number', type=int, default=0)
    #parser.add_argument('--temp_folders', action='store_true', help='reuse one temp folder/name and retain only essential data to save space')
    parser.add_argument('--natedebug', action='store_true', help='activate vscode debugger')
    parser.add_argument('--run_prefix', type=str, default='test')
    parser.add_argument('--res', type=int, default=4)
    parser.add_argument('--sparsity', type=int, default=8)
    #parser.add_argument('--demo', action = 'store_true', help = 'Use to avoid merging and actually creating folders')
    #parser.add_argument('--early_stop', type=str, default = 'full_run', help='Set this to any stage you want to halt at, otherwise will run entire GO pipeline')

    args = parser.parse_args()
    
    if args.natedebug:
        import debugpy
        debugpy.listen(5678)
        debugpy.wait_for_client() 
    custom_run(args)