import re
import sys

downsample = 'python preprocess/downsample.py -s "$data_dir"'

preprocessing = 'python preprocess/pred_monodepth.py -s "$data_dir"'

visual = 'python visual_hull.py --sparse_id $sparsity --data_dir "$data_dir" --reso $resolution --not_vis'

first_train = 'python train_gs.py -s "$data_dir" -m output/gs_init/"$dir_name" \
	-r $resolution --sparse_view_num $sparsity --sh_degree 2 \
	--init_pcd_name visual_hull_$sparsity \
	--white_background --random_background'

first_render = 'python render.py -m output/gs_init/"$dir_name" \
	--sparse_view_num $sparsity --sh_degree 2 \
	--init_pcd_name visual_hull_$sparsity \
	--white_background --skip_all --skip_train'

first_render_cont = 'python render.py -m output/gs_init/"$dir_name" \
	--sparse_view_num $sparsity --sh_degree 2 \
	--init_pcd_name visual_hull_$sparsity \
	--white_background --render_path'

loo1 = 'python leave_one_out_stage1.py -s "$data_dir" \
	-m output/gs_init/"$dir_name"_loo \
	-r $resolution --sparse_view_num $sparsity --sh_degree 2 \
	--init_pcd_name visual_hull_$sparsity \
	--white_background --random_background'

loo2 = 'python leave_one_out_stage2.py -s "$data_dir" \
	-m output/gs_init/"$dir_name"_loo \
	-r $resolution --sparse_view_num $sparsity --sh_degree 2 \
	--init_pcd_name visual_hull_$sparsity \
	--white_background --random_background'

lora = 'python train_lora.py --exp_name controlnet_finetune/"$dir_name" \
	--prompt xxy5syt00 --sh_degree 2 --resolution $resolution --sparse_num $sparsity \
	--data_dir "$data_dir" \
	--gs_dir output/gs_init/"$dir_name" \
	--loo_dir output/gs_init/"$dir_name"_loo \
	--bg_white --sd_locked --train_lora --use_prompt_list \
	--add_diffusion_lora --add_control_lora --add_clip_lora'

repair_train = 'python train_repair.py \
    --config configs/gaussian-object.yaml \
    --train --gpu 0 \
    tag="$dir_name" \
    system.init_dreamer="output/gs_init/"$dir_name"" \
    system.exp_name="output/controlnet_finetune/"$dir_name"" \
    system.refresh_size=8 \
    data.data_dir="$data_dir" \
    data.resolution=$resolution \
    data.sparse_num=$sparsity \
    data.prompt="a photo of a xxy5syt00" \
    data.refresh_size=8 \
    system.sh_degree=2'

final_render = 'python render.py \
    -m output/gs_init/"$dir_name" \
    --sparse_view_num $sparsity --sh_degree 2 \
    --init_pcd_name visual_hull_$sparsity \
    --white_background --skip_all --skip_train \
    --load_ply output/gaussian_object/"$dir_name"/save/last.ply'

final_render_cont = 'python render.py \
    -m output/gs_init/"$dir_name" \
    --sparse_view_num $sparsity --sh_degree 2 \
    --init_pcd_name visual_hull_$sparsity \
    --white_background --render_path \
    --load_ply output/gaussian_object/"$dir_name"/save/last.ply'

pred_poses = 'python pred_poses.py -s "$data_dir" --sparse_num $sparsity'

progs = {
    "downsample" : downsample,
    "preprocessing" : preprocessing,
    "visual" : visual,
    "first_train" : first_train,
    "first_render" : first_render,
    "first_render_cont" : first_render_cont,
    "loo1" : loo1,
    "loo2" : loo2,
    "lora" : lora,
    "repair_train" : repair_train,
    "final_render" : final_render,
    "final_render_cont" : final_render_cont,
    "pred_poses" : pred_poses
}

allCmds = [downsample, preprocessing, visual, first_train, first_render, first_render_cont, loo1, loo2, lora, repair_train, final_render, final_render_cont]

def whitespace_remover(varString):
    components = re.split('\s+', varString)
    return ' '.join(components)

def var_replacer(varString, dirName, dataDir, sparseNum, resolution):
    newString = varString.replace('"$dir_name"', dirName)
    newString = newString.replace('"$data_dir"', dataDir)
    newString = newString.replace("$sparsity", sparseNum)
    newString = newString.replace("$resolution", resolution)
    
    return newString

def dust3r_replacer(newString, dirName):
    if newString.startswith('python pred_poses.py'):
        return newString
    if newString.startswith('python visual_hull.py') or newString.startswith('python preprocess'):
        newString = ''
        return newString
    if newString.startswith('python train_repair.py ') or ('--load_ply' in newString):
        if newString.startswith('python train_repair.py'):
            newString = newString + f' data.json_path="output/gs_init/{dirName}/refined_cams.json"'
            newString = newString.replace('configs/gaussian-object.yaml', 'configs/gaussian-object-colmap-free.yaml')
        elif '--load_ply' in newString:
            newString = newString + ' --use_dust3r'
    else:
        if not newString.startswith('python train'):
            newString = newString + f' --dust3r_json output/gs_init/{dirName}/refined_cams.json'
        newString = newString + ' --use_dust3r'
    newString = newString.replace('visual_hull', 'dust3r')
    return newString


if __name__ == '__main__':
    """
    args:
    1 - Which prompt to return, by name; if 'all', returns all at once
    2 - dirName, run instance name (e.g. firsttest-frame_0sec)
    3 - dataDir; path to run instance (e.g. data/natedata/dirName)
    4 - sparsity number
    5 - Using dust3r (Any # > 0 is True)
    6 - Resolution (default 1)
    7 - Using VSCode debugger (default 0)
    """
    #print('got args:\n')
    #for i in range(6):
    #    print(sys.argv[i])
    #    print(type(sys.argv[i]))
    #print('end got args')

    if sys.argv[1] in progs.keys():
        temp = whitespace_remover(progs[sys.argv[1]])
        final = var_replacer(temp, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[6])
        if int(sys.argv[5]) > 0:
            final = dust3r_replacer(final, sys.argv[2])
        if int(sys.argv[7]) == 1 and not 'render' in sys.argv[1]:
            final += ' --natedebug'
        print(final)
    elif sys.argv[1] == 'all':
        print(f'\n\n\033[45mStarting command printout:\033[0m \n')
        if int(sys.argv[5]) > 0: # The use dust3r case
            temp = whitespace_remover(pred_poses)
            final = var_replacer(temp, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[6])
            print('\n')
            print(final)
        for cmd in allCmds:
            temp = whitespace_remover(cmd)
            final = var_replacer(temp, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[6])
            if int(sys.argv[5]) > 0:
                final = dust3r_replacer(final, sys.argv[2])
            if not 'render' in cmd and int(sys.argv[7]) == 1:
                final += ' --natedebug'
            print('\n')
            print(final)