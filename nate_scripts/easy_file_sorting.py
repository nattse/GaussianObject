import os
import re
import shutil

srcDir = '/Volumes/ExtremeSSD/nerf_recordings/more_nerf_recordings/25-8-30_allgopros/frames_of_interest'
dstDir = '/Volumes/ExtremeSSD/nerf_recordings/more_nerf_recordings/25-8-30_allgopros/biggestMerge'
files = os.listdir(srcDir)
for camnum in range(1,9): 
    camfiles = [i for i in files if i.startswith(f'G{camnum}')]
    jpgs = [i for i in camfiles if i.endswith('.jpg')]
    pngs = [i for i in camfiles if i.endswith('.png')]
    jpgs = sorted(jpgs, key = lambda x: int(re.search('frame(\d+)(.jpg|.png)', x)[1]))
    pngs = sorted(pngs, key=lambda x: int(re.search('frame(\d+)(.jpg|.png)', x)[1]))
    print(f'Length of files: {len(jpgs)} before pruning')
    #jpgs = jpgs[::2]
    #pngs = pngs[::2]
    #print(f'Length of files: {len(jpgs)} after pruning')
    if not os.path.isdir(f'{dstDir}/cam{camnum}'):
        os.mkdir(f'{dstDir}/cam{camnum}')
    if not os.path.isdir(f'{dstDir}2/cam{camnum}'):
        os.mkdir(f'{dstDir}2/cam{camnum}')
    for f in jpgs[::2]:
        shutil.copy(os.path.join(srcDir, f), f'{dstDir}/cam{camnum}')
    for f in jpgs[1::2]:
        shutil.copy(os.path.join(srcDir, f), f'{dstDir}2/cam{camnum}')
    for f in pngs[::2]:
        shutil.copy(os.path.join(srcDir, f), f'{dstDir}/cam{camnum}')
    for f in pngs[1::2]:
        shutil.copy(os.path.join(srcDir, f), f'{dstDir}2/cam{camnum}')
    
