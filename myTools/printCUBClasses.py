import os.path as osp

root = r'F:\birdsDetection\CUB_200_2011\CUB_200_2011'
out_dir = r'F:\birdsDetection\yolov5-master\datasets\cub-200-2001'

with open(osp.join(root, 'classes.txt'), 'r') as f:
    lines = f.readlines()
with open(osp.join(out_dir, 'classes.txt'), 'w') as f:
    for line in lines:
        idx, name = line.split()
        f.write(f'{int(idx)-1}: {name}\n')
print('done!')