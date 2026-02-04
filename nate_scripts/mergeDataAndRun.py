import os
import shutil
import re
import toml
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import subprocess
import argparse

class MergedFolders:
    """
        For processing whole videos, we split frame images into cam subfolders for easier handling.
        We also store masks in these same subfolders. This function makes all processes easier by gathering
        all masks/images into a single object. Subfolers must be named 'cam1', 'cam2' etc.
        Actual images must be named '*camX*frameY.jpg'
        :param rootDir: Where the subfolders are, created by frame_ripper.py
    """
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.filePattern = False
        subdirs = [i for i in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, i)) and 'cam' in i]
        subdirs = sorted(subdirs, key = lambda x: re.search('cam([0-9]+)', x)[1])
        print(f'Are these the camera subdirs (and are they in the right order)?\n{rootDir}:')
        for subdir in subdirs:
            print(subdir)
        #approve = input('y/n')
        #if approve.lower() != 'y':
        #    return
        dfMaster = []
        for subdir in subdirs:
            mbImages = [i.split('.')[0] for i in os.listdir(os.path.join(rootDir, subdir)) if i.endswith('.jpg') and not i.startswith('.') and 'frame' in i]
            mbMasks = [i.split('.')[0] for i in os.listdir(os.path.join(rootDir, subdir)) if i.endswith('.png') and not i.startswith('.') and 'frame' in i]
            common = list(set(mbImages).intersection(mbMasks))
            if self.filePattern == False: # Used to extract the file pattern
                self.filePattern = re.search('(.+cam|G)[0-9]+(.+frame)[0-9]+', common[0]).groups()
            indices = [int(re.search('frame([0-9]+)', i)[1]) for i in common]
            df = pd.DataFrame(common, index = indices)
            df.columns = [subdir.split('cam')[-1]]
            df = df.sort_index()
            dfMaster.append(df)
        # Whenever we try to index a video frame from our dataset, we check this df first to make sure every camera has the image and mask for that frame
        self.dfMaster = ~pd.concat(dfMaster, axis = 1).isna()

    def __getitem__(self, index):
        if not self.dfMaster.loc[index].all():
            return None
        filePaths = {}
        temp_imgs = []
        temp_msks = []
        for i in self.dfMaster.columns:
            filename = self.rootDir + '/' + f'cam{i}' + '/' + self.filePattern[0] + i + self.filePattern[1] + str(index)
            temp_imgs.append(filename + '.jpg')
            temp_msks.append(filename + '.png')
        filePaths['images'] = temp_imgs
        filePaths['masks'] = temp_msks
        return filePaths

    def __iter__(self):
        valid = self.dfMaster.index[self.dfMaster.all(axis=1)]
        return iter(valid)

    def __len__(self):
        valid = self.dfMaster.index[self.dfMaster.all(axis=1)]
        return len(valid)


def build_frame_folder(buildDir, filePaths, projectName, calibFile):
    project = os.path.join(buildDir, projectName)
    if os.path.isdir(project):
        shutil.rmtree(project)
    os.mkdir(project)
    projIm = os.path.join(project, 'images')
    os.mkdir(projIm)
    projMask = os.path.join(project, 'masks')
    os.mkdir(projMask)
    projS = os.path.join(project, 'sparse', '0')
    os.makedirs(projS)
    for count, src in enumerate(filePaths['images']):
        shutil.copy(src, projIm + '/' + f'{count}.jpg')
    for count, src in enumerate(filePaths['masks']):
        shutil.copy(src, projMask + '/' + f'{count}.png')

    camIntrinsics, camExtrinsics, points3D = colmap_like(len(filePaths['images']), calibFile)

    with open(os.path.join(projS, 'cameras.txt'), 'w') as file:
        file.writelines(camIntrinsics)
    with open(os.path.join(projS, 'images.txt'), 'w') as file:
        file.writelines(camExtrinsics)
    with open(os.path.join(projS, 'points3D.txt'), 'w') as file:
        file.writelines(points3D)

    with open(os.path.join(project, 'sparse_8.txt'), 'w') as file:
        file.writelines(['0\n', '1\n', '2\n', '3\n', '4\n', '5\n', '6\n', '7'])
    with open(os.path.join(project, 'sparse_test.txt'), 'w') as file:
        file.writelines(['0\n', '1'])



def colmap_like(numCams, calibFile):
    an_cal = toml.load(calibFile)
    camIntrinsics = []
    camExtrinsics = []
    points3D = []
    for count in range(numCams):
        cam_num = f'cam_{count}'  # So named because that's just how the calib.yaml for avian3d is set up
        cam_cal = an_cal[cam_num]

        # Intrinsics: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS
        # Note: Must use Pinhole camera model, which as paramter list fx, fy, cx, cy
        matrix = np.array(cam_cal['matrix'])
        fx, fy = matrix[[0, 1], [0, 1]]
        cx, cy = matrix[[0, 1], [2, 2]]
        w, h = cam_cal['size']

        intrinsics = f'{count + 1} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n'
        camIntrinsics.append(intrinsics)

        # Extrinsics must be listed as two rows:
        # 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        # 2: POINTS2D[] as (X, Y, POINT3D_ID) (which we will fill in with dummy points)
        rot = np.array(cam_cal['rotation'])
        rotMat = R.from_euler('xyz', rot).as_matrix()
        trans = np.array(cam_cal['translation'])
        mat = np.eye(4, 4)
        mat[:3, :3] = rotMat
        mat[:3, 3] = trans
        correctMat = opencv_to_opengl(mat)
        rotMat = correctMat[:3, :3]

        # QX, QY, QZ, QW = R.from_euler('xyz', rot).as_quat() # Returns scalar last
        QX, QY, QZ, QW = R.from_matrix(rotMat).as_quat()  # Returns scalar last
        TX, TY, TZ = correctMat[:3, 3]

        extrinsics = f'{count + 1} {QW} {QX} {QY} {QZ} {TX} {TY} {TZ} {count + 1} {count}.jpg\n'
        extrinsics_2 = '0 0 -1\n'
        camExtrinsics.append(extrinsics)
        camExtrinsics.append(extrinsics_2)

        # Points3D: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
        # But we just fill in totally dummy data since there's no point in making these points all match up and whatnot
        points = f'{count + 1} 0.90221661188998425 2.6738268646791883 0.30519660713850205 165 168 173 0.0059492972328126885 171 74 157 85\n'
        points3D.append(points)
    return camIntrinsics, camExtrinsics, points3D


def opencv_to_opengl(mat):
    """
    Makes it so that Z points in the direction of camera's view, while ensuring that
    objects on "left" and "right" of a view are the same before and after the conversion.
    Previously we had used mat * np.array([1,-1,-1]) to flip the y and z axis, which produced
    camera poses that seemed correct, but this results in left and right being switched.
    Do not use if importing from Blender, which already uses OpenGL convention.
    (Colmap technically needs to convert to OpenGL style but that code uses the left/right
    flip way, not sure why that works fine)

    input:
    mat = 4x4 extrinsics matrix
    returns the same
    """
    mat_copy = mat.copy()
    # Rotate 180 deg around Z-axis
    theta = np.pi
    r_mat = np.array([
        [np.cos(theta), -1 * np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    mat_ = mat[:3, :3]
    mat_ = mat_ @ r_mat
    mat_copy[:3, :3] = mat_ * np.array([1, -1, -1])
    return mat_copy



def repeatedly_run(args):
    """
    Merges images to form temp folders, run through GO, move the output files, then overwrite temp folder
    for next run. 
    """
    bashScript = f'./run_data.sh'
    runName = f'run{args.input_data}_res{args.res}'
    input_data = f'{args.rootDir}/data/{args.input_data}'
    calibFile = [i for i in os.listdir(input_data) if 'calibration' in i and i.endswith('toml') and not i.startswith('.')]
    calibFile = os.path.join(input_data, calibFile[0])

    if os.path.expanduser('~') == '/home/leelab':
        outputDir = './output/nate_outputs'
    else:
        rootDir = '/Volumes/ExtremeSSD/samGUI/export'
        outputDir = '/Users/nathanieltse/Desktop/gaussianObject/onlygopros/project/export'
        bashScript = None
        calibFile = '/Users/nathanieltse/Desktop/gaussianObject/onlygopros/project/calibration/calibration.toml'
    if args.early_stop == 'full_run':
        runName = runName + '_fullrun'
        cmdArgs = [
            "-r", f'{args.run_prefix}',
            "-p", f'{runName}',
            "-s", str(args.sparsity),
            "-z", str(args.res),
            "-o"
        ]
        outputData = f'./output/gaussian_object/{args.run_prefix}-{runName}/save/last.ply'
        outputRender = f'./output/gs_init/{args.run_prefix}-{runName}/render/ours_None/renders.mp4'
    else:
        cmdArgs = [
            "-r", f'{args.run_prefix}',
            "-p", f'{runName}',
            "-s", str(args.sparsity),
            "-z", str(args.res),
            "-X", args.early_stop, #"first_render",
            "-o"
        ]
        outputData = f'./output/gs_init/{args.run_prefix}-{runName}/point_cloud/iteration_10000/point_cloud.ply'
        outputRender = f'./output/gs_init/{args.run_prefix}-{runName}/render/ours_10000/renders.mp4'
    merge = MergedFolders(input_data)
    if len(args.framerange) > 0:
        start, stop = int(args.framerange[0]), int(args.framerange[1])
    else:
        start, stop = None, None
    for i in merge:
        if not start is None:
            if i < start or i > stop:
                continue
        build_frame_folder(buildDir = '/home/leelab/Desktop/GaussianObject/data/natedata',
                           filePaths = merge[i],
                           projectName=f'{runName}',
                           calibFile = calibFile)
        cmd = ["bash", bashScript] + cmdArgs
        subprocess.run(cmd)
        #proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if os.path.isfile(outputData):
            shutil.copy(outputData, os.path.join(outputDir, f'{runName}-frame{i}.ply'))
            #shutil.copy(outputRender, os.path.join(outputDir, f'{runName}-frame{i}.mp4'))

def save_repeatedly(args):
    """
    Like repeatedly_run, except we don't form temp folders and ensure outputs of each individual run are 
    fully retained.
    """
    bashScript = f'./run_data.sh'
    #_runName = f'run{args.input_data}_res{args.res}'
    _runName = args.run_prefix
    input_data = f'{args.rootDir}/data/{args.input_data}'
    calibFile = [i for i in os.listdir(input_data) if 'calibration' in i and i.endswith('toml') and not i.startswith('.')]
    calibFile = os.path.join(input_data, calibFile[0])

    if not os.path.expanduser('~') == '/home/leelab':
        rootDir = '/Volumes/ExtremeSSD/samGUI/export'
        outputDir = '/Users/nathanieltse/Desktop/gaussianObject/onlygopros/project/export'
        bashScript = None
        calibFile = '/Users/nathanieltse/Desktop/gaussianObject/onlygopros/project/calibration/calibration.toml'
    merge = MergedFolders(input_data)
    if len(args.framerange) > 0:
        start, stop = int(args.framerange[0]), int(args.framerange[1])
    else:
        start, stop = None, None
    for i in merge:
        if not start is None:
            if i < start or i > stop:
                continue
        runName = _runName + f'_frame{i}'
        if args.early_stop == 'full_run':
            runName += '_fullrun'
            cmdArgs = [
                "-r", f'{args.run_prefix}',
                "-p", f'{runName}',
                "-s", str(args.sparsity),
                "-z", str(args.res),
                "-o",
            ]
        else:
            cmdArgs = [
                "-r", f'{args.run_prefix}',
                "-p", f'{runName}',
                "-s", str(args.sparsity),
                "-z", str(args.res),
                "-X", args.early_stop,
                "-o",
            ]
        cmd = ["bash", bashScript] + cmdArgs
        print(cmd)
        if not args.demo:
            build_frame_folder(buildDir = '/home/leelab/Desktop/GaussianObject/data/natedata',
                            filePaths = merge[i],
                            projectName=f'{runName}',
                            calibFile = calibFile)
        
            subprocess.run(cmd)
            #subprocess.Popen(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create GO project folders on the fly, either temporarily to save space or uniquely named to investigate.')
    parser.add_argument('--input_data', type=str, default='toMerge', help='dir in ../data to process')
    parser.add_argument('--temp_folders', action='store_true', help='reuse one temp folder/name and retain only essential data to save space')
    parser.add_argument('--natedebug', action='store_true', help='activate vscode debugger')
    parser.add_argument('--run_prefix', type=str, default='test')
    parser.add_argument('--res', type=int, default=4)
    parser.add_argument('--sparsity', type=int, default=8)
    parser.add_argument('--demo', action = 'store_true', help = 'Use to avoid merging and actually creating folders')
    parser.add_argument('--early_stop', type=str, default = 'full_run', help='Set this to any stage you want to halt at, otherwise will run entire GO pipeline')
    parser.add_argument('--rootDir', default = '/home/leelab/Desktop/GaussianObject')
    parser.add_argument('--framerange', default=[], nargs="+", type=str, help = 'start and stop frames if we do not want to process all frames in the merge dir')
    args = parser.parse_args()
    
    if args.natedebug:
        # If things dont shut down gracefully, this port may still be left still alive,
        # so find with lsof -i :5678 and then kill the PID
        import debugpy
        debugpy.listen(5678)
        debugpy.wait_for_client() 
    
    os.chdir(args.rootDir)
    
    if args.temp_folders:
        repeatedly_run(args)
    else:
        save_repeatedly(args)
    
