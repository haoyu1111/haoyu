- [Abstract](#abstract)
- [Project Flow](#project-flow)
  * [Print CUB-200 dataset Classes](#print-cub-200-dataset-classes)
  * [Convert image format to yolo format](#convert-image-format-to-yolo-format)
    + [Yolo format](#yolo-format)
- [Comparing the performence of different Yolo models](#comparing-the-performence-of-different-yolo-models)
  * [Yolov5s (Baseline)](#yolov5s--baseline-)
  * [Yolov5l](#yolov5l)
  * [Yolov8](#yolov8)
  * [Using TTA trick](#using-tta-trick)
  * [Apply DIOU-NMS](#apply-diou-nms)
  * [Performences for different models](#performences-for-different-models)
- [Finally using Yolov5L+DIOU-NMS+TTA model train the NEW Mandai bird park dataset.](#finally-using-yolov5l-diou-nms-tta-model-train-the-new-mandai-bird-park-dataset)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## Abstract




This project implements a bird classification task based on the Yolov5 deep
learning model for the CUB-200 dataset and Mandai Bird Park and compares the
performance of the different models.


## Project Flow


### Print CUB-200 dataset Classes

[Run:printCUBClasses.py](myTools/printCUBClasses.py)


```python
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
```

### Convert image format to yolo format
#### Yolo format
Labels for this format should be exported to YOLO format with one file per image. If there are no objects in an image, no file is required. The file should be formatted with one row per object in format. Box coordinates must be in normalized xywh format (from 0 to 1). If your boxes are in pixels, you should divide and by image width, and and by image height. Class numbers should be zero-indexed (start with 0).
![在这里插入图片描述](https://img-blog.csdnimg.cn/473107e4d3634671af3bbd3ec577b3e7.png)
[Run:convertCUB2YOLOFormat.py](myTools/convertCUB2YOLOFormat.py)
 Here is the code for changing the format of CUB-200 dataset to Yolo format:
```python
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
```
The Ultralytics YOLO format is a dataset configuration format that allows you to define the dataset root directory, the relative paths to training/validation/testing image directories or *.txt files containing image paths, and a dictionary of class names. Here is the example:

```python

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/cub-200-2001  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/val  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
names:
  0: 001.Black_footed_Albatross
  1: 002.Laysan_Albatross
  2: 003.Sooty_Albatross
  3: 004.Groove_billed_Ani
  4: 005.Crested_Auklet
  ...
  199: 200.Common_Yellowthroat
```

## Comparing the performence of different Yolo models 
### Yolov5s (Baseline)
For using Yolov5s model to train, just need modify the Environment variables of "train.py". For example:
Environment variables:
data represents the dataset used for training.
cfg represents the trained model.
weights represents the pre-trained weights.

```python
--data
cub200.yaml
--cfg
yolov5s.yaml
--batch-size
8
--img
640
--epochs
100
--weights
yolov5s.pt
--workers
0
```
### Yolov5l

Just need to change the cfg to "yolov5l.yaml" and change the weights to the "yolov5l.pt".
### Yolov8

Also try the nest version of Yolo model. But the performence is mediocre.
### Using TTA trick
A simple introduction to Test Time Augmentation (TTA) is to use data augmentation during testing as well.

Implementation in yolov5:
Append --augment to any existing val.py command to enable TTA, and increase the image size by about 30% for improved results. For example:
 `python val.py --weights yolov5l.pt --img 832 --source data/images --augment
`
Note: The weights here are the best performing weights in a training session.

### Apply DIOU-NMS
DIOU-NMS considers not only the IOU, but also the distance between the centre points of the two frames. If the IOU between two frames is relatively large, but when the distance between two frames is relatively large, it may be considered as a frame of two objects and not be filtered out. 
 
 In the YOLOV5 source code, the author is directly calling the official Pytorch NMS API. 
In the non_max_suppression function in general.py

```python
i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
```
Replace with DIOU-NMS function：

```python
 
   def diou_box_nms(self, scores, boxes, iou_thres):
        if boxes.shape[0] == 0:
            return torch.zeros(0,device=boxes.device).long()
        x1,y1,x2,y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:,3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = torch.sort(scores, descending=True)[1] #(?,)
        keep =[]
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            else:
                i = order[0].item()
                keep.append(i)
 
                xmin = torch.clamp(x1[order[1:]], min = float(x1[i]))
                ymin = torch.clamp(y1[order[1:]], min = float(y1[i]))
                xmax = torch.clamp(x2[order[1:]], max = float(x2[i]))
                ymax = torch.clamp(y2[order[1:]], max = float(y2[i]))
 
                inter_area = torch.clamp(xmax - xmin, min=0.0) * torch.clamp(ymax - ymin, min=0.0)
 
                iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-16)
 
                # diou add center
                # inter_diag
                cxpreds = (x2[i] + x1[i]) / 2
                cypreds = (y2[i] + y1[i]) / 2
 
                cxbbox = (x2[order[1:]] + x1[order[1:]]) / 2
                cybbox = (y1[order[1:]] + y2[order[1:]]) / 2
 
                inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2
 
                # outer_diag
                ox1 = torch.min(x1[order[1:]], x1[i])
                oy1 = torch.min(y1[order[1:]], y1[i])
                ox2 = torch.max(x2[order[1:]], x2[i])
                oy2 = torch.max(y2[order[1:]], y2[i])
 
                outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2
 
                diou = iou - inter_diag / outer_diag
                diou = torch.clamp(diou, min=-1.0, max=1.0)
 
 
                # mask_ind = (iou <= iou_thres).nonzero().squeeze()
                mask_ind = (diou <= iou_thres).nonzero().squeeze()
 
                if mask_ind.numel() == 0:
                    break
                order = order[mask_ind + 1]
        return torch.LongTensor(keep)
```

### Performences for different models
![在这里插入图片描述](https://img-blog.csdnimg.cn/2262b1b4ac394784be137508da6c83b7.png)

By comparing the above tricks I used, yolov5L with TTA as well as DIOU tricks  performs best.


## Finally using Yolov5L+DIOU-NMS+TTA model train the NEW Mandai bird park dataset.
Every bird in the New Mandai Bird Park dataset was detected correctly and with a high degree of confidence. And The goal to detecting species of birds in the New Mandai bird park was finally achieved. 
