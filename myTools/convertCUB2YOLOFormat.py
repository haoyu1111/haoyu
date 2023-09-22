import os
import os.path as osp

from PIL import Image, ImageDraw
from tqdm import tqdm
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default=r'C:\Users\DELL\Desktop\ee5003\datasets\CUB_200_2011')
parser.add_argument('--out_dir', type=str, default=r'C:\Users\DELL\Desktop\ee5003\datasets\cub-200-2001')
args = parser.parse_args()

# create out dirs
if osp.isdir(args.out_dir):
    shutil.rmtree(args.out_dir)
os.makedirs(osp.join(args.out_dir, 'images', 'train'), exist_ok=True)
os.makedirs(osp.join(args.out_dir, 'images', 'val'), exist_ok=True)
os.makedirs(osp.join(args.out_dir, 'labels', 'train'), exist_ok=True)
os.makedirs(osp.join(args.out_dir, 'labels', 'val'), exist_ok=True)

with open(osp.join(args.input_dir, 'images.txt'), 'r') as f:
    images_txt = f.readlines()
with open(osp.join(args.input_dir, 'train_test_split.txt'), 'r') as f:
    splits_txt = f.readlines()
with open(osp.join(args.input_dir, 'bounding_boxes.txt'), 'r') as f:
    bboxes_txt = f.readlines()
assert len(images_txt) == len(splits_txt) == len(bboxes_txt)

for img_line, split_line, bbox_line in tqdm(zip(images_txt, splits_txt, bboxes_txt), total=len(images_txt)):
    img_id, img_name = img_line.split()
    img_id1, is_training = split_line.split()
    img_id2, x, y, w, h = bbox_line.split()
    cls = int(osp.basename(osp.dirname(img_name)).split('.')[0]) - 1
    x, y, w, h = list(map(float, [x, y, w, h]))
    assert img_id1 == img_id == img_id2 and img_name.endswith('.jpg') and 0<=cls<=199
    if is_training == '1':
        mode = 'train'
    else:
        mode = 'val'

    # copy image to yolo format dir
    img_name = osp.join(args.input_dir, 'images', img_name)
    assert osp.isfile(img_name)
    img = Image.open(img_name)
    height, width = img.height, img.width
    out_img_name = osp.join(args.out_dir, 'images', mode, f'{cls+1}.{osp.basename(img_name)}')
    img.save(out_img_name)

    # show bbox
    # draw = ImageDraw.ImageDraw(img)
    # draw.rectangle([int(x), int(y), int(x+w), int(y+h)], outline=(255, 0, 0))
    # img.show()
    # convert annotation to yolo format
    x_center, y_center, w, h = (x+w/2)/width, (y+h/2)/height, w/width, h/height

    with open(out_img_name.replace('images', 'labels').replace('.jpg', '.txt'), 'w') as f:
        f.write(f'{str(cls)} {str(x_center)} {str(y_center)} {str(w)} {str(h)}')
print('done!')