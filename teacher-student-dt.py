import torchvision
import ultralytics.engine.results
from ultralytics import YOLO
import torch
import os
import random
import shutil
import numpy as np
import labels

home_path = lambda x: os.path.join('.', x)
scratch_path = lambda x: os.path.join('.', x)

class_dict = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6,
              'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13,
              'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}
reverse_class_dict = {v: k for k, v in class_dict.items()}

threshold = 'dynamic_simple'  # 'static', 'dynamic_simple', 'dynamic'
class_weights = {0: 0., 1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0., 10: 0., 11: 0., 12: 0., 13: 0.,
                 14: 0.,
                 15: 0., 16: 0., 17: 0., 18: 0., 19: 0.}

ensemble = True
max_wh = 7680


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
    for name in os.listdir(scratch_path('data/VOC/ss-1/images/')):
        shutil.move(scratch_path('data/VOC/ss-1/images/' + name),
                    scratch_path('data/VOC/train/images/' + name))
        os.remove(scratch_path('data/VOC/ss-1/labels/' + name.replace('.jpg', '.txt')))
    for name in os.listdir(scratch_path('data/VOC/ss-2/images/')):
        shutil.move(scratch_path('data/VOC/ss-2/images/' + name),
                    scratch_path('data/VOC/train/images/' + name))
        os.remove(scratch_path('data/VOC/ss-2/labels/' + name.replace('.jpg', '.txt')))

    if os.path.isfile(scratch_path('data/VOC/semi-supervised/labels.cache')):
        os.remove(scratch_path('data/VOC/semi-supervised/labels.cache'))
    if os.path.isfile(scratch_path('data/VOC/ss-1/labels.cache')):
        os.remove(scratch_path('data/VOC/ss-1/labels.cache'))
    if os.path.isfile(scratch_path('data/VOC/ss-2/labels.cache')):
        os.remove(scratch_path('data/VOC/ss-2/labels.cache'))


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


def ensemble_iteration(teacher, label, data):
    # Train student model
    print("New base weights: {}".format(os.path.join(teacher.trainer.save_dir, 'weights', 'last.pt')))
    student = YOLO(os.path.join(teacher.trainer.save_dir, 'weights', 'last.pt'))
    student.train(data=data, epochs=1, device='cpu', workers=0, pretrained=True,
                  project=home_path('runs/train'),
                  name=label)  # Change save dir project and name, where save_dir=project/name
    return student


def partition_arrays(label_dict):
    # Convert dictionary to list of arrays and list of keys
    keys = list(label_dict.keys())
    arrays = np.array(list(label_dict.values()))

    # Calculate the total sum of all arrays
    total_sum = np.sum(arrays, axis=0)
    num_classes = len(total_sum)

    # Initialize sums for the two partitions
    sum1 = np.zeros(num_classes)
    sum2 = np.zeros(num_classes)

    # Partitions
    partition1 = []
    partition2 = []

    # Sort arrays based on their total counts in descending order
    sorted_indices = np.argsort(-np.sum(arrays, axis=1))

    for idx in sorted_indices:
        array = arrays[idx]
        key = keys[idx]
        # Determine which partition to add the current array to
        if np.sum(np.abs((sum1 + array) - sum2)) < np.sum(np.abs(sum1 - (sum2 + array))):
            sum1 += array
            partition1.append(key)
        else:
            sum2 += array
            partition2.append(key)

    return partition1, partition2


# Reset data
reset()
# torch.cuda.set_device(0)

if ensemble:
    # Load labels
    u = np.unique(np.array([v for v in labels.ts_10.values()]))
    ls = {}

    for label in u:
        with open(scratch_path('data/VOC/train/labels/' + label + '.txt'), 'r') as file:
            cnt = [0 for _ in range(20)]
            for c in file:
                cnt[int(c.split()[0])] += 1
            ls[label] = cnt

    # Partition labels
    p1, p2 = partition_arrays(ls)

    # Move partitions to respective folders
    for label in p1:
        shutil.move(scratch_path('data/VOC/train/images/' + label + '.jpg'),
                    scratch_path('data/VOC/ss-1/images/' + label + '.jpg'))
        shutil.copy(scratch_path('data/VOC/train/labels/' + label + '.txt'),
                    scratch_path('data/VOC/ss-1/labels/' + label + '.txt'))

    for label in p2:
        shutil.move(scratch_path('data/VOC/train/images/' + label + '.jpg'),
                    scratch_path('data/VOC/ss-2/images/' + label + '.jpg'))
        shutil.copy(scratch_path('data/VOC/train/labels/' + label + '.txt'),
                    scratch_path('data/VOC/ss-2/labels/' + label + '.txt'))

    # Train teachers
    t1 = YOLO('yolov8n.pt')
    t1.train(data='VOC1.yaml', epochs=1, device='cpu', workers=0, project=home_path('runs/train'),
             name='teacher_1')  # Change save dir project and name, where save_dir=project/name
    t2 = YOLO('yolov8n.pt')
    t2.train(data='VOC2.yaml', epochs=1, device='cpu', workers=0, project=home_path('runs/train'),
             name='teacher_2')  # Change save dir project and name, where save_dir=project/name

    # Iteratively assign pseudo-labels and train student model
    n = 3
    for k in range(n):
        if len(os.listdir(scratch_path('data/VOC/train/images'))) == 0:
            break
        r1 = static_threshold(t1)
        r2 = static_threshold(t2)

        # Combine results from both teachers
        r = zip(r1, r2)
        added = 0
        size = len(os.listdir(scratch_path('data/VOC/train/images')))
        for i, (x, y) in enumerate(r, start=1):
            assert x.path == y.path
            print('processing ', i, '/', size, ' ', x.path)
            l = os.path.basename(x.path).replace('.jpg', '')

            cls = torch.cat((x.boxes.cls, y.boxes.cls), dim=0)
            boxes = torch.cat((x.boxes.xyxy, y.boxes.xyxy), dim=0)
            conf = torch.cat((x.boxes.conf, y.boxes.conf), dim=0)
            bwhn = torch.cat((x.boxes.xywhn, y.boxes.xywhn), dim=0)

            bnms = boxes[:, :4] + cls[:, None] * max_wh
            idx = torchvision.ops.nms(bnms, conf, 0.7)
            res = torch.cat((cls.unsqueeze(1), bwhn), dim=1)
            res = res[idx]
            if res.shape[0] > 0:
                added += 1
                cnt = [0 for _ in range(20)]
                with open(scratch_path('data/VOC/semi-supervised/labels/' + l + '.txt'),
                          'w') as file:
                    for t in res:
                        cnt[int(t[0].item())] += 1
                        line = tuple(t.tolist())
                        file.write(("%g " * len(line)).rstrip() % line)
                        file.write('\n')
                shutil.move(scratch_path('data/VOC/train/images/' + l + '.jpg'),
                            scratch_path('data/VOC/semi-supervised/images/' + l + '.jpg'))
                ls[l] = cnt

        # Partition labels
        for label in os.listdir(scratch_path('data/VOC/ss-1/images')):
            label = label.replace('.jpg', '')
            shutil.move(scratch_path('data/VOC/ss-1/images/' + label + '.jpg'),
                        scratch_path('data/VOC/semi-supervised/images/' + label + '.jpg'))
            shutil.move(scratch_path('data/VOC/ss-1/labels/' + label + '.txt'),
                        scratch_path('data/VOC/semi-supervised/labels/' + label + '.txt'))
        for label in os.listdir(scratch_path('data/VOC/ss-2/images')):
            label = label.replace('.jpg', '')
            shutil.move(scratch_path('data/VOC/ss-2/images/' + label + '.jpg'),
                        scratch_path('data/VOC/semi-supervised/images/' + label + '.jpg'))
            shutil.move(scratch_path('data/VOC/ss-2/labels/' + label + '.txt'),
                        scratch_path('data/VOC/semi-supervised/labels/' + label + '.txt'))

        p1, p2 = partition_arrays(ls)

        for label in p1:
            shutil.move(scratch_path('data/VOC/semi-supervised/images/' + label + '.jpg'),
                        scratch_path('data/VOC/ss-1/images/' + label + '.jpg'))
            shutil.move(scratch_path('data/VOC/semi-supervised/labels/' + label + '.txt'),
                        scratch_path('data/VOC/ss-1/labels/' + label + '.txt'))
        for label in p2:
            shutil.move(scratch_path('data/VOC/semi-supervised/images/' + label + '.jpg'),
                        scratch_path('data/VOC/ss-2/images/' + label + '.jpg'))
            shutil.move(scratch_path('data/VOC/semi-supervised/labels/' + label + '.txt'),
                        scratch_path('data/VOC/ss-2/labels/' + label + '.txt'))

        print(added, ' images passed the threshold and were added to the training sets.')

        # Train iteration
        t1 = ensemble_iteration(t1, 'student'+str(k+1)+'_1', 'VOC1.yaml')
        t2 = ensemble_iteration(t2, 'student'+str(k+1)+'_2', 'VOC2.yaml')

else:
    # Setup data subset for semi-supervised learning
    setup(570)  # 5717 training images total, 570 - approx. 10%, 1140 - approx. 20%, 2850 - approx. 50%.
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
