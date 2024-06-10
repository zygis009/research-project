import xml.etree.ElementTree as ET
import os
import shutil


def convert_xml_to_yolo(xml_file_path, txt_file_path, class_dict):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Retrieve image dimensions
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # create empty txt file
    open(txt_file_path, 'w').close()

    # Open the output file
    with open(txt_file_path, 'w') as file:
        # Iterate over each object in the XML
        for obj in root.iter('object'):
            class_name = obj.find('name').text
            # Convert class name to a class ID based on provided dictionary
            class_id = class_dict.get(class_name, -1)
            if class_id == -1:
                continue  # Skip if class is not found

            # Parse bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = int(round(float(bndbox.find('xmin').text)))
            ymin = int(round(float(bndbox.find('ymin').text)))
            xmax = int(round(float(bndbox.find('xmax').text)))
            ymax = int(round(float(bndbox.find('ymax').text)))

            # Calculate YOLO format coordinates
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            # Write to file
            file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")


# Example usage

##classes  person
# • bird, cat, cow, dog, horse, sheep
# • aeroplane, bicycle, boat, bus, car, motorbike, train
# • bottle, chair, dining table, potted plant, sofa, tv/monitor
class_dict = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6,
              'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13,
              'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}

scratch_path = lambda x: os.path.join('.', x)

# Clean up data directory
if os.path.exists(scratch_path('data/VOC')):
    shutil.rmtree(scratch_path('data/VOC'))
if os.path.exists(scratch_path('data/VOCtrainval_11-May-2012')):
    shutil.rmtree(scratch_path('data/VOCtrainval_11-May-2012'))

# Uncompress the VOC dataset
shutil.unpack_archive(scratch_path('data/VOCtrainval_11-May-2012.tar'), scratch_path('data/VOCtrainval_11-May-2012'),
                      'tar')

# Create directories for VOC dataset
try:
    os.makedirs(scratch_path('data/VOC/AllLabels'))
    os.makedirs(scratch_path('data/VOC/semi-supervised/images'))
    os.mkdir(scratch_path('data/VOC/semi-supervised/labels'))
    os.makedirs(scratch_path('data/VOC/test/images'))
    os.mkdir(scratch_path('data/VOC/test/labels'))
    os.makedirs(scratch_path('data/VOC/train/images'))
    os.mkdir(scratch_path('data/VOC/train/labels'))
    os.makedirs(scratch_path('data/VOC/val/images'))
    os.mkdir(scratch_path('data/VOC/val/labels'))
    os.makedirs(scratch_path('data/VOC/ss-1/images'))
    os.mkdir(scratch_path('data/VOC/ss-1/labels'))
    os.makedirs(scratch_path('data/VOC/ss-2/images'))
    os.mkdir(scratch_path('data/VOC/ss-2/labels'))
except OSError as error:
    print(error)

# Loop through all the xml files in the directory:

for file in os.listdir(scratch_path('data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations')):
    if file.endswith('.xml'):
        xml_file_path = os.path.join(scratch_path('data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations'), file)
        txt_file_path = os.path.join(scratch_path('data/VOC/AllLabels'), file.replace('.xml', '.txt'))
        convert_xml_to_yolo(xml_file_path, txt_file_path, class_dict)

# Split the data into train and validation

train = []
val = []
for f in open(scratch_path('data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/train.txt')):
    train.append(f.strip())

for f in open(scratch_path('data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/val.txt')):
    val.append(f.strip())

# Move training labels and images to the train folder
for name in train:
    src_label = os.path.join(scratch_path('data/VOC/AllLabels'), name + '.txt')
    dst_label = os.path.join(scratch_path('data/VOC/train/labels'), name + '.txt')
    shutil.move(src_label, dst_label)
    src_img = os.path.join(scratch_path('data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages'), name + '.jpg')
    dst_img = os.path.join(scratch_path('data/VOC/train/images'), name + '.jpg')
    shutil.move(src_img, dst_img)

# Move validation labels and images to the val folder
for name in val:
    src_label = os.path.join(scratch_path('data/VOC/AllLabels'), name + '.txt')
    dst_label = os.path.join(scratch_path('data/VOC/val/labels'), name + '.txt')
    shutil.move(src_label, dst_label)
    src_img = os.path.join(scratch_path('data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages'), name + '.jpg')
    dst_img = os.path.join(scratch_path('data/VOC/val/images'), name + '.jpg')
    shutil.move(src_img, dst_img)

# Move remaining labels and images to the test folder
for file in os.listdir(scratch_path('data/VOC/AllLabels')):
    src_label = os.path.join(scratch_path('data/VOC/AllLabels'), file)
    dst_label = os.path.join(scratch_path('data/VOC/test/labels'), file)
    shutil.move(src_label, dst_label)
    src_img = os.path.join(scratch_path('data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages'),
                           file.replace('.txt', '.jpg'))
    dst_img = os.path.join(scratch_path('data/VOC/test/images'), file.replace('.txt', '.jpg'))
    shutil.move(src_img, dst_img)

# Clean up
os.rmdir(scratch_path('data/VOC/AllLabels'))
