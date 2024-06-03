import ultralytics.engine.results
from ultralytics import YOLO
import torch
import os
import random
import shutil
import numpy as np

home_path = lambda x: os.path.join('/home/zliutkus', x)
scratch_path = lambda x: os.path.join('/scratch/zliutkus/research-project', x)

class_dict = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6,
              'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13,
              'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}
reverse_class_dict = {v: k for k, v in class_dict.items()}

threshold = 'dynamic_simple'  # 'static', 'dynamic_simple', 'dynamic'
class_weights = {0: 0., 1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0., 10: 0., 11: 0., 12: 0., 13: 0.,
                 14: 0.,
                 15: 0., 16: 0., 17: 0., 18: 0., 19: 0.}


def setup(size=250):
    with open(scratch_path('data/voc-classes.txt'), 'r') as file:
        classes = file.readlines()
    split = size // len(classes)
    used = set()
    for class_name in classes:
        class_name = class_name.strip()
        with open(scratch_path(
                os.path.join('data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/',
                             class_name + '_train.txt')),
                'r') as file0:
            names = [x for x in file0.readlines() if x.split()[1] != '-1']
        picked = random.sample(names, split)
        print(class_name, ": ", picked)
        for name in picked:
            name = name.split()[0]
            if name in used:
                continue

            used.add(name)
            shutil.move(scratch_path('data/VOC/train/images/' + name + '.jpg'),
                        scratch_path('data/VOC/semi-supervised/images/' + name + '.jpg'))
            shutil.copy(scratch_path('data/VOC/train/labels/' + name + '.txt'),
                        scratch_path('data/VOC/semi-supervised/labels/' + name + '.txt'))

            with open(scratch_path('data/VOC/semi-supervised/labels/' + name + '.txt'), 'r') as file1:
                boxes = file1.readlines()
            for box in boxes:
                class_weights[int(box.split()[0])] += 1


def reset():
    for name in os.listdir(scratch_path('data/VOC/semi-supervised/images/')):
        shutil.move(scratch_path('data/VOC/semi-supervised/images/' + name),
                    scratch_path('data/VOC/train/images/' + name))
        os.remove(scratch_path('data/VOC/semi-supervised/labels/' + name.replace('.jpg', '.txt')))

    if os.path.isfile(scratch_path('data/VOC/semi-supervised/labels.cache')):
        os.remove(scratch_path('data/VOC/semi-supervised/labels.cache'))


def static_threshold(model, conf=0.95):
    return model.predict(source=scratch_path('data/VOC/train/images'), stream=True, conf=conf, device='cpu')


def dynamic_threshold_simple(model, conf_min=0.8, conf_max=1.0):
    fg_count = sum(class_weights.values())
    print("Before: ")
    for key in class_weights:
        print("Class: ", reverse_class_dict[key], " Weight: ", class_weights[key], " Threshold: ",
              get_threshold(class_weights[key] / fg_count, min_threshold=conf_min, max_threshold=conf_max))
    print("Total: ", fg_count)
    results = model.predict(source=scratch_path('data/VOC/train/images'), device='cpu')

    for result in results:
        filtered_boxes_data = []
        for box in zip(result.boxes.cls, result.boxes.conf, result.boxes.data):
            cls_ratio = class_weights[int(box[0].item())] / fg_count
            conf_threshold = get_threshold(cls_ratio, min_threshold=conf_min, max_threshold=conf_max)
            if box[1].item() >= conf_threshold:
                filtered_boxes_data.append(box)
                class_weights[int(box[0].item())] += 1
                fg_count += 1

        if filtered_boxes_data:
            filtered_boxes = ultralytics.engine.results.Boxes(torch.stack([box[2] for box in filtered_boxes_data]),
                                                              result.boxes.orig_shape)
        else:
            filtered_boxes = ultralytics.engine.results.Boxes(torch.empty((0, 6)), result.boxes.orig_shape)

        result.boxes = filtered_boxes
    print("After: ")
    for key in class_weights:
        print("Class: ", reverse_class_dict[key], " Weight: ", class_weights[key], " Threshold: ",
              get_threshold(class_weights[key] / fg_count, min_threshold=conf_min, max_threshold=conf_max))
    print("Total: ", fg_count)

    return results


def dynamic_threshold(model, conf_thresh=0.95, gamma=0.05):
    results = model.predict(source=scratch_path('data/VOC/train/images'))
    fg_count = 0
    for key in class_weights:
        class_weights[key] = 0

    for result in results:
        boxes = [(int(cls.item()), conf.item()) for cls, conf in zip(result.boxes.cls, result.boxes.conf)]
        for cls, conf in boxes:
            class_weights[cls] += conf
        fg_count += len(boxes)

    avg_fg_count = fg_count / len(class_dict.keys())
    for key in class_weights:
        print("Class: ", reverse_class_dict[key], " Weight: ", class_weights[key], " Threshold: ",
              np.power(class_weights[key] / avg_fg_count, gamma) * conf_thresh)
    print("Confidence: ", conf_thresh)
    for result in results:
        filtered_boxes_data = []
        for box in zip(result.boxes.cls, result.boxes.conf, result.boxes.data):
            cls_ratio = np.power(class_weights[int(box[0].item())] / avg_fg_count, gamma)
            if box[1].item() >= cls_ratio * conf_thresh:
                filtered_boxes_data.append(box)

        if filtered_boxes_data:
            filtered_boxes = ultralytics.engine.results.Boxes(torch.stack([box[2] for box in filtered_boxes_data]),
                                                              result.boxes.orig_shape)
        else:
            filtered_boxes = ultralytics.engine.results.Boxes(torch.empty((0, 6)), result.boxes.orig_shape)

        result.boxes = filtered_boxes
    return results


def get_threshold(ratio, mid_ratio=1 / 20, min_threshold=0.8, mid_threshold=0.95, max_threshold=1.0):
    # Compute the base of the exponential function for each segment
    base_left = np.power(mid_threshold / min_threshold, 1 / mid_ratio)
    base_right = np.power(max_threshold / mid_threshold, 1 / (1 - mid_ratio))

    # Compute the threshold using the exponential function
    if ratio <= mid_ratio:
        threshold = min_threshold * np.power(base_left, ratio)
    else:
        threshold = mid_threshold * np.power(base_right, 1 - ratio ** -1 * mid_ratio)

    return threshold


def student_iteration(teacher):
    # Train student model
    print("New base weights: {}".format(os.path.join(teacher.trainer.save_dir, 'weights', 'last.pt')))
    student = YOLO(os.path.join(teacher.trainer.save_dir, 'weights', 'last.pt'))
    student.train(data='VOC.yaml', epochs=100, device=0, workers=0, pretrained=True,
                  project=home_path('runs/train'),
                  name='train_dt_10_')  # Change save dir project and name, where save_dir=project/name
    return student


# Setup data subset for semi-supervised learning
reset()
setup(570)  # 5717 training images total, 570 - approx. 10%, 1140 - approx. 20%, 2850 - approx. 50%.
torch.cuda.set_device(0)

# Train teacher model
teacher = YOLO('yolov8n.pt')
teacher.train(data='VOC.yaml', epochs=100, device=0, workers=0, project=home_path('runs/train'),
              name='train_dt_10_')  # Change save dir project and name, where save_dir=project/name

# Iteratively assign pseudo-labels and train student model
n = 3
for _ in range(n):
    if len(os.listdir(scratch_path('data/VOC/train/images'))) == 0:
        break
    results = static_threshold(teacher) if threshold == 'static' else dynamic_threshold_simple(
        teacher) if threshold == 'dynamic_simple' else dynamic_threshold(teacher)
    added = 0
    for result in results:
        if result.boxes.shape[0] > 0:
            added = added + 1
            result.save_txt(scratch_path(
                'data/VOC/semi-supervised/labels/' + os.path.basename(result.path).replace('.jpg', '.txt')))
            shutil.move(scratch_path('data/VOC/train/images/' + os.path.basename(result.path)),
                        scratch_path('data/VOC/semi-supervised/images/' + os.path.basename(result.path)))
    print(added, ' images passed the threshold and were added to the training set.')
    teacher = student_iteration(teacher)
