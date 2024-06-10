import os

import torch
import torchvision
from ultralytics import YOLO
from ultralytics.utils import ops
import shutil
import numpy as np
import matplotlib.pyplot as plt
import csv


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


def adjust_gumbel_softmax(params, t=1.0):
    logs = torch.log(params)
    noise = -torch.log(-torch.log(torch.rand(params.size())))
    gumbels = logs + noise
    # return torch.nn.functional.softmax(gumbels / t, dim=-1)

    return torch.nn.functional.gumbel_softmax(torch.log(params), tau=t, hard=False)


def sigmoid(x, scale=10, max_value=1.0, threshold=0.25):
    return (1 / max_value - 1) * 1 / (1 + np.exp(-scale * (x - threshold))) + 1


def adjust_values(values, threshold=0.25):
    above_threshold = values >= threshold

    # Compute max and second max
    max_values, _ = torch.max(values, dim=-1)
    snd_max_values = torch.topk(values, 2, dim=-1)[0][..., -1]

    # Compute the mask for max values
    max_mask = values == max_values.unsqueeze(-1)
    max_mask = max_mask & above_threshold

    # Compute the difference between max and second max
    dif = max_values - snd_max_values

    # Select relevant dif and max_values
    count = max_mask.count_nonzero(dim=-1)
    indices = count.nonzero(as_tuple=True)
    dif = dif[indices].repeat_interleave(count[indices])
    max_values = max_values[indices].repeat_interleave(count[indices])

    # Apply the sigmoid function to the max values
    values[max_mask] *= sigmoid(dif, max_value=max_values, scale=200.0, threshold=0.25)

    # # Update the above_threshold mask to exclude max values
    # above_threshold = above_threshold & ~max_mask
    #
    # # Apply the exponential function to the values above the threshold
    # for i in range(values.shape[1]):
    #     values[:, i][above_threshold[:, i]] *= torch.exp(-1 * (max_values[:, i][above_threshold[:, i].any(dim=-1)] - values[:, i][above_threshold[:, i]]))
    return values


def adjust_values_old(values, threshold=0.25):
    above_threshold = values >= threshold
    if np.count_nonzero(above_threshold) < 1:
        return values
    max = np.max(values[above_threshold])
    max_values = values == max
    snd_max = np.partition(values, -2)[-2]
    above_threshold = (values >= threshold) & (values != max)
    if max != snd_max:
        dif = max - snd_max
        values[max_values] *= sigmoid(dif, max_value=max, scale=20, threshold=0.25)
    values[above_threshold] *= np.exp(-1 * (max - values[above_threshold]))
    return values


def find_equal_tensors(tensors):
    tensor_dict = {}
    equal_pairs = []

    for i, tensor in enumerate(tensors):
        # Convert tensor to a hashable type
        tensor_hash = hash(tensor.numpy().tobytes())

        if tensor_hash in tensor_dict:
            # If the tensor is already in the dictionary, add the pair of indices to the result
            for j in tensor_dict[tensor_hash]:
                equal_pairs.append((j, i))
            tensor_dict[tensor_hash].append(i)
        else:
            # Otherwise, add the tensor to the dictionary
            tensor_dict[tensor_hash] = [i]

    return equal_pairs


def read_results_csv(path):
    xs = []
    ys50 = []
    ys50_95 = []
    with open(path, 'r') as file:
        dict = csv.reader(file)
        for i, row in enumerate(dict):
            if i == 0:
                continue
            # print('Epoch: ', row[0].strip(), ' mAP50: ', row[6].strip(), ' mAP50_95: ', row[7].strip())
            xs.append(int(row[0].strip()))
            ys50.append(float(row[6].strip()))
            ys50_95.append(float(row[7].strip()))
    return ys50, ys50_95


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


def read_class_counts(path):
    counts = {v: 0 for v in range(20)}
    with open(path, 'r') as file:
        for line in file:
            cnts = line.split(maxsplit=4)[4].replace('{', '').replace('}', '').split(', ')
            for cnt in cnts:
                cnt = cnt.split(': ')
                counts[int(cnt[0])] += int(cnt[1].strip())
    return counts


def read_total_object_counts(path):
    total = []
    with open(path, 'r') as file:
        for line in file:
            total.append(int(line.split()[1]))
    return total


def read_total_image_counts(path):
    total = []
    with open(path, 'r') as file:
        initial = [next(file) for _ in range(20)]
        names = []
        for line in initial:
            names.extend(
                [x.split()[0] for x in line.split(maxsplit=2)[2].replace('[', '').replace(']', '').split(', ')])
        total.append(len(np.unique(names)))
        for line in file:
            if line.endswith('added to the training set.\n'):
                total.append(int(line.split()[0]))
    return total


"""
Practise ensemble pseudo-labeling example
"""

m1 = YOLO('dt/train_dt_10_1/weights/best.pt')
m2 = YOLO('dt/train_dt_10_2/weights/best.pt')

img = 'bus.jpg'
r1 = m1.predict(source=img)
r2 = m2.predict(source=img)

max_wh = 7680
cls1 = [r.boxes.cls for r in r1]
boxes1 = [r.boxes.xyxy for r in r1]
conf1 = [r.boxes.conf for r in r1]
bwhn1 = [r.boxes.xywhn for r in r1]
cls2 = [r.boxes.cls for r in r2]
boxes2 = [r.boxes.xyxy for r in r2]
conf2 = [r.boxes.conf for r in r2]
bwhn2 = [r.boxes.xywhn for r in r2]

cls = torch.cat((torch.cat(cls1, dim=0), torch.cat(cls2, dim=0)), dim=0)
boxes = torch.cat((torch.cat(boxes1, dim=0), torch.cat(boxes2, dim=0)), dim=0)
conf = torch.cat((torch.cat(conf1, dim=0), torch.cat(conf2, dim=0)), dim=0)
bwhn = torch.cat((torch.cat(bwhn1, dim=0), torch.cat(bwhn2, dim=0)), dim=0)

res = torch.cat((cls.unsqueeze(1), bwhn), dim=1)

boxes_nms = boxes[:, :4] + cls[:, None] * max_wh
i = torchvision.ops.nms(boxes_nms, conf, 0.7)
res = res[i]
with open('eg-res.txt', 'w') as file:
    for t in res:
        line = tuple(t.tolist())
        file.write(("%g " * len(line)).rstrip() % line)
        file.write('\n')

"""
Teacher-Student vs Confidence Scaling
"""
# fig, ax = plt.subplots()
# x = ['Teacher', '1st Student', '2nd Student', '3rd Student']
# handles = []
# for i in ['10', '20', '50']:
#     ts = read_total_object_counts('added_counts/ts-' + i + '-added.log')
#     # ts = read_total_image_counts('outs_logs/ts-' + i + '.log')
#     cs = read_total_object_counts('added_counts/cs-' + i + '-added.log')
#     # cs = read_total_image_counts('colab_results/cs_' + i + '.out.log')
#     if i == '10':
#         ax.plot(np.cumsum(ts), label='Naive Teacher-Student', color='tab:blue')
#         ax.plot(np.cumsum(cs), label='Confidence Scaling', color='tab:orange')
#     else:
#         ax.plot(np.cumsum(ts), color='tab:blue')
#         ax.plot(np.cumsum(cs), color='tab:orange')
#     ax.annotate(i+'%', xy=(-0.13, ts[0]))
# ax.set_xticks(range(4))
# ax.set_xticklabels(x, rotation=45)
# ax.set_xlabel('Iteration')
# ax.set_ylabel('Object Count')
# ax.set_title('Cumulative Object Counts')
# plt.legend()
# plt.tight_layout()
# plt.show()

"""
Image count - object count correlation plot
"""
# for i in ['10', '20', '50']:
#     objects_ts = read_total_object_counts('added_counts/ts-' + i + '-added.log')
#     images_ts = read_total_image_counts('outs_logs/ts-' + i + '.log')
#     objects_dt = read_total_object_counts('added_counts/dt-' + i + '-added.log')
#     if i == '50':
#         images_dt = read_total_image_counts('outs_logs/dt-' + i + '.log')
#     else:
#         images_dt = read_total_image_counts('colab_results/dt_' + i + '.out.log')
#     fig, ax = plt.subplots()
#     ax.plot(np.cumsum(images_ts), np.cumsum(objects_ts), label='Naive Teacher-Student')
#     ax.plot(np.cumsum(images_dt), np.cumsum(objects_dt), label='Dynamic Thresholds')
#     ax.set_xlabel('Image Count')
#     ax.set_ylabel('Object Count')
#     ax.set_title('Image Count - Object Count Correlation ' + i + '%')
#     # ax.set_xlim([0, 6000])
#     # ax.set_ylim([0, 10000])
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

"""
Plotting class count barchart
"""
# class_names = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car',
#                'motorbike', 'train', 'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor']
# for i in ['10', '20', '50']:
#     counts = read_class_counts('added_counts/cs_dt-' + i + '-added.log')
#     print('{total: ', sum(counts.values()), ', counts: ', counts.values(), '}')
#     fig, ax = plt.subplots()
#     ax.bar(counts.keys(), counts.values())
#     ax.set_xticks(range(20))
#     ax.set_xticklabels(class_names, rotation=90)
#     ax.set_ylim([0, 5100])
#     ax.set_xlabel('Classes')
#     ax.set_ylabel('Object Count')
#     ax.set_title('Final object counts: Confidence scaling + Dynamic thresholds ' + i + '%')
#     plt.tight_layout()
#     plt.show()
#     # plt.savefig('plots/counts/cs_dt_' + i + '_bar.png')

"""
Extracting the class counts from the logs
"""

# class_dict = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6,
#               'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13,
#               'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}
#
# counts = {v: 0 for v in class_dict.values()}

# with open('colab_results/cs_50.out.log', 'r') as file:
#     for line in file:
#         if line.startswith('image'):
#             cnts = line.split(maxsplit=6)[6].split(', ')[:-1]
#             if cnts[0] != '(no detections)':
#                 for cnt in cnts:
#                     cnt = cnt.split(maxsplit=1)
#                     counts[class_dict[cnt[1] if cnt[0] == '1' else cnt[1][:-1]]] += int(cnt[0])
#         if line.startswith('New base weights'):
#             print('{total: ', sum(counts.values()), ', counts: ', counts, '}')
#             counts = {v: 0 for v in class_dict.values()}


# with open('outs_logs/dt-50.log', 'r') as file:
#     flag = -1
#     for line in file:
#         if line.startswith('Before:'):
#             flag = -1
#         if line.startswith('After:'):
#             flag = 1
#         if line.startswith('Class:'):
#             cnt = line.split()
#             counts[class_dict[cnt[1]]] += int(float(cnt[3])) * flag
#         if line.startswith('New base weights'):
#             print('{total: ', sum(counts.values()), ', counts: ', counts, '}')
#             counts = {v: 0 for v in class_dict.values()}

# with open('colab_results/cs_dt_50.out.log', 'r') as file:
#     prev_counts = {v: 0 for v in class_dict.values()}
#     for line in file:
#         if line.startswith('Class:'):
#             cnt = line.split()
#             counts[class_dict[cnt[1]]] = int(float(cnt[3])) - prev_counts[class_dict[cnt[1]]]
#             prev_counts[class_dict[cnt[1]]] = int(float(cnt[3]))
#         if line.startswith('Total:'):
#             print('{total: ', sum(counts.values()), ', counts: ', counts, '}')

# with open('colab_results/dt_20.out.log', 'r') as file:
#     flag = False
#     for line in file:
#         if 'train_dt_20_3' in line:
#             flag = True
#         if flag and line.startswith('Class:'):
#             cnt = line.split()
#             counts[class_dict[cnt[1]]] = int(float(cnt[3]))
#
#
# with open('temp.txt', 'r') as file:
#     for line in file:
#         if line.startswith('Class:'):
#             cnt = line.split()
#             counts[class_dict[cnt[1]]] = int(float(cnt[3])) - counts[class_dict[cnt[1]]]
#
# print('{total: ', sum(counts.values()), ', counts: ', counts, '}')

# model = YOLO('yolov8n.pt')
# # model.val(data='VOC.yaml')
# results = model.predict(
#     source='bus.jpg',
#     stream=False,)
# # results[0].save('results_bus1.jpg')
# print(results[0].boxes)

"""
Plotting the class counts
"""
# Data
# categories = ['Naive\nTeacher-Student', 'Dynamic\nThresholds', 'Confidence\nScaling',
#               'Confidence Scaling\n+\nDynamic Thresholds']
# subcategories = ['Teacher', '1st Student', '2nd Student', '3rd Student']
# # Objects
# stats = {
#     '10%': [[1829, 2739, 172, 114], [1942, 2021, 249, 247], [1815, 10216, 435, 92], [1818, 10897, 355, 41]],
#     '20%': [[3732, 1973, 109, 108], [3579, 1872, 242, 109], [3706, 9619, 288, 47], [3692, 9913, 253, 44]],
#     '50%': [[8231, 929, 97, 84], [7970, 1360, 117, 82], [8315, 6813, 105, 18], [8123, 6777, 126, 21]]
# }
# # Images
# # stats = {
# #     '10%': [[541, 2081, 172, 113], [553, 1755, 245, 244], [551, 4550, 365, 82], [544, 4713, 293, 36]],
# #     '20%': [[1097, 1612, 109, 106], [1104, 1635, 238, 108], [1095, 4229, 243, 42], [1094, 4281, 205, 37]],
# #     '50%': [[2553, 802, 97, 84], [2558, 1238, 116, 81], [2573, 2986, 88, 12], [2576, 2980, 101, 19]]
# # }
#
# # Params
# width = 0.25
# padding = 0.75
# x = np.arange(len(categories)) * (len(categories) * width + padding)
#
# # Plot
# fig, ax = plt.subplots()
# colors = ['y', 'g', 'b', 'r']
#
# handles = []
# for i, (name, data) in enumerate(stats.items()):
#     bottom = np.zeros(len(categories))
#     for j in range(len(subcategories)):
#         bars = ax.bar(x + i * width, [data[k][j] for k in range(len(categories))], width, bottom=bottom,
#                       color=colors[j])
#         if i == 0:
#             handles.append(bars[0])
#         bottom += [data[k][j] for k in range(len(categories))]
#
# ax.set_xlabel('Method')
# ax.set_ylabel('Added Images')
# ax.set_title('Added Images per Method')
# ax.set_xticks(x + (len(categories) - 1) * width / 2)
# ax.set_xticklabels(categories)
#
# ax.legend(handles, subcategories, title='Iterations')
# plt.tight_layout()
# plt.savefig('plots/counts/added_images.png')

"""
Plotting added training image counts
"""
# ts10 = [541, 2081, 172, 113]
# ts20 = [1097, 1612, 109, 106]
# ts50 = [2553, 802, 97, 84]
# dt10 = [553, 1755, 245, 244]
# dt20 = [1104, 1635, 238, 108]
# dt50 = [2558, 1238, 116, 81]
# cs10 = [551, 4550, 365, 82]
# cs20 = [1095, 4229, 243, 42]
# cs50 = [2573, 2986, 88, 12]
# cs_dt10 = [544, 4713, 293, 36]
# cs_dt20 = [1094, 4281, 205, 37]
# cs_dt50 = [2576, 2980, 101, 19]

# xs = ['Initial', '1st Iteration', '2nd Iteration', '3rd Iteration']
#
# f, ax = plt.subplots(1)
# # ax.plot(xs, np.cumsum(ts10), label='Teacher-Student')
# # ax.plot(xs, np.cumsum(ts20), label='Teacher-Student')
# ax.plot(xs, np.cumsum(ts50), label='Teacher-Student')
# # ax.plot(xs, np.cumsum(dt10), label='Dynamic Thresholds')
# # ax.plot(xs, np.cumsum(dt20), label='Dynamic Thresholds')
# ax.plot(xs, np.cumsum(dt50), label='Dynamic Thresholds')
# ax.set_xticks(xs)
# ax.set_xticklabels(xs, rotation=45)
# ax.set_xlabel('Iteration')
# ax.set_ylabel('Training Image Count')
# ax.legend()
# ax.set_title('Cumulative Image Counts 50%')
# f.tight_layout()
# f.savefig('plots/cumulative_image_counts_50.png')


"""
Plotting the threshold function
"""

# xs = np.linspace(0, 1, 1000)
# ys = [get_threshold(x) for x in xs]
# plt.plot(xs, ys)
# plt.xlabel('Class Ratio')
# plt.ylabel('Threshold')
# plt.show()

"""
Plotting the sigmoid and adjust values functions for different parameters
"""

# for i in [x / 100.0 for x in range(5, 50, 10)]:
#     xs = [np.array(a) for a in zip(np.linspace(0, 1, 100), i * np.ones(100))]
#     ys = [adjust_values(torch.Tensor([[x]]))[0][0][0] for x in xs]
#     plt.plot(np.linspace(0, 1, 100), ys, label='Second Max: ' + str(i))
#
# # for max_val in [0.25, 0.5, 0.75, 0.85, 0.95]:
# #     plt.plot(np.linspace(0, 1, 100), sigmoid(np.linspace(0, 1, 100), max_value=max_val, scale=10, threshold=0.45),
# #              label='Max Value: ' + str(max_val))
# plt.plot([0, 1], [1, 1], color='black', linestyle='dashed')
# plt.legend()
# plt.show()

"""
Bar chart for adjusted example confidence values (gumbel-softmax)
"""

# heights = torch.Tensor(
#     [6.4535e-04, 1.2104e-03, 4.4408e-01, 3.4658e-04, 1.7597e-01, 5.1181e-05, 2.5829e-03, 1.4453e-04, 2.3322e-03,
#      1.1503e-02, 1.7264e-02, 1.3000e-04, 6.1941e-03, 2.4574e-05, 5.9322e-03, 3.7143e-03, 4.4532e-04, 1.1711e-02,
#      7.7523e-04, 2.4549e-05])
#
# classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
#            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
#            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
# # [1.2989e-03, 1.5494e-03, 4.3264e-03, 4.1190e-05, 2.2488e-02, 1.0156e-04, 1.3433e-03, 6.1275e-04, 1.1476e-03,
# #        1.3656e-03, 1.0394e-03, 1.4070e-04, 1.1044e-04, 9.4954e-05, 1.3522e-01, 6.3556e-02, 1.1106e-03, 3.5448e-01,
# #        6.2345e-03, 7.4728e-03]

# heights = np.array(heights)
# heights = adjust_gumbel_softmax(heights)  # Uncomment to plot the adjusted values
#
# fig, ax = plt.subplots()
#
# ax.bar(classes, heights)
# ax.set(xlabel='Classes', ylabel='Confidence')
#
# ax.set_xticks(range(len(classes)))
# ax.set_xticklabels(classes, rotation=90)
# plt.ylim([0, 1])
# plt.show()

"""
Plotting the mAP results of the dynamic threshold and teacher-student experiments
"""

# # Due to a changed NMS function, the cs and cs_dt results are not valid, thus validation
# # was redone with an unchanged NMS function
# data = {
#     'cs': {10: {'mAP50': [0.573165526992492, 0.5952304789002284, 0.5882500383051139, 0.5833085781518926],
#                 'mAP50-95': [0.38448175176055444, 0.41078077542583885, 0.4045730488143018, 0.39945588803870585]},
#            20: {'mAP50': [0.6243620828262674, 0.629013257162516, 0.6184583139588484, 0.614466034304872],
#                 'mAP50-95': [0.42757092676409164, 0.43252600620202436, 0.4265507328839376, 0.42238288206387226]},
#            50: {'mAP50': [0.669517515347205, 0.6682895087462393, 0.6615066841102004, 0.6596646575654495],
#                 'mAP50-95': [0.4729659578689534, 0.47499442024011146, 0.4671800279987147, 0.4648249543928499]}
#            },
#     'cs_dt': {10: {'mAP50': [0.5947546770020142, 0.6075508485315467, 0.5982294225985016, 0.5924497072097299],
#                    'mAP50-95': [0.39911641590986396, 0.41931094855283246, 0.41166501869959865, 0.4066363019151146]},
#               20: {'mAP50': [0.631294326123829, 0.6373362489174068, 0.6275982866390146, 0.6164106961720439],
#                    'mAP50-95': [0.43549995633881716, 0.443047683900428, 0.4344307864826412, 0.4255074063329218]},
#               50: {'mAP50': [0.6711270187114082, 0.6691508948556132, 0.6621263839149235, 0.6519775073370658],
#                    'mAP50-95': [0.47608235560557943, 0.4721795266343773, 0.4654457220453079, 0.45936602450112396]}
#               }
# }
#
# name_dict = {1: 'teacher', 2: '1st student', 3: '2nd student', 4: '3rd student'}
# f1, ax1 = plt.subplots(1)
# f2, ax2 = plt.subplots(1)
# ax1.set_title('Teacher-Student mAP50')
# ax2.set_title('Teacher-Student mAP50-95')
#
# for i in [10, 20, 50]:
#     xs = ['Teacher', '1st Student', '2nd Student', '3rd Student']
#     ys50 = []  # data['cs'][i]['mAP50']
#     ys50_95 = []  # data['cs'][i]['mAP50-95']
#     for j in [1, 2, 3, 4]:
#         path = 'ts/ts_' + str(i) + '_' + str(j) + '/results.csv'
#         ys1, ys2 = read_results_csv(path)
#         ys50.append(ys1[-1])
#         ys50_95.append(ys2[-1])
#
#     ax1.plot(xs, ys50, label=str(i) + '% labeled')
#     ax2.plot(xs, ys50_95, label=str(i) + '% labeled')
#
# b1, b2 = read_results_csv('baseline/results.csv')
# ax1.axhline(y=b1[-1], color='black', linestyle='dashed', label='Baseline')
# ax2.axhline(y=b2[-1], color='black', linestyle='dashed', label='Baseline')
# ax1.set_xticks(xs)
# ax1.set_xticklabels(xs, rotation=45)
# ax2.set_xticks(xs)
# ax2.set_xticklabels(xs, rotation=45)
# ax1.set_xlabel('Iteration')
# ax1.set_ylabel('mAP50')
# ax2.set_xlabel('Iteration')
# ax2.set_ylabel('mAP50-95')
# # title = 'Dynamic thresholds ' + str(i) + '%'
# # ax1.set_title(title)
# # ax2.set_title(title)
# ax1.set_ylim([0.49, 0.72])
# ax2.set_ylim([0.34, 0.53])
# ax1.legend()
# ax2.legend()
# f1.tight_layout()
# f2.tight_layout()
# # f1.show()
# # f2.show()
# f1.savefig('plots/ts_mAP50_last.png')
# f2.savefig('plots/ts_mAP50_95_last.png')

"""
Plotting mAP per iteration for the teacher-student experiment
"""

# name_dict = {1: 'Teacher', 2: '1st Student', 3: '2nd Student', 4: '3rd Student'}
# xs = np.linspace(0, 400, 400)
# for i in [10, 20, 50]:
#     y50 = []
#     y50_95 = []
#     for j in [1, 2, 3, 4]:
#         path = 'ts/ts_' + str(i) + '_' + str(j) + '/results.csv'
#         ys50, ys50_95 = read_results_csv(path)
#         y50.extend(ys50)
#         y50_95.extend(ys50_95)
#     f1, ax1 = plt.subplots(1)
#     f2, ax2 = plt.subplots(1)
#     ax1.set_title('mAP50 ' + str(i) + '%')
#     ax2.set_title('mAP50:95:5 ' + str(i) + '%')
#     ax1.plot(xs, y50)
#     ax2.plot(xs, y50_95)
#     ax1.set_xticks([50, 150, 250, 350])
#     ax1.set_xticklabels(name_dict.values())
#     ax1.set_xlabel('Iterations')
#     ax1.set_ylabel('mAP')
#     ax1.set_xlim([-5, 405])
#     ax1.set_ylim([0.1, 0.7])
#     ax2.set_xticks([50, 150, 250, 350])
#     ax2.set_xticklabels(name_dict.values())
#     ax2.set_xlabel('Iterations')
#     ax2.set_ylabel('mAP')
#     ax2.set_xlim([-5, 405])
#     ax2.set_ylim([0.05, 0.5])
#     # Draw vertical lines at the boundaries of the intervals
#     for pos in [100, 200, 300]:
#         ax1.axvline(pos, color='gray', linestyle='--', linewidth=0.8)
#         ax2.axvline(pos, color='gray', linestyle='--', linewidth=0.8)
#
#     # Shade the intervals for better visual separation
#     ax1.axvspan(-5, 100, color='blue', alpha=0.1)
#     ax1.axvspan(100, 200, color='green', alpha=0.1)
#     ax1.axvspan(200, 300, color='red', alpha=0.1)
#     ax1.axvspan(300, 405, color='yellow', alpha=0.1)
#     ax2.axvspan(-5, 100, color='blue', alpha=0.1)
#     ax2.axvspan(100, 200, color='green', alpha=0.1)
#     ax2.axvspan(200, 300, color='red', alpha=0.1)
#     ax2.axvspan(300, 405, color='yellow', alpha=0.1)
#
#     # Add grid for better readability
#     ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
#     ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
#
#     f1.tight_layout()
#     f2.tight_layout()
#     f1.savefig('plots/mAP_per_epoch/ts_mAP50_' + str(i) + '.png')
#     f2.savefig('plots/mAP_per_epoch/ts_mAP50_95_' + str(i) + '.png')
