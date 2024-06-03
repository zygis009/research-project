from ultralytics import YOLO
import os
import random
import shutil

home_path = lambda x: os.path.join('/home/zliutkus', x)
scratch_path = lambda x: os.path.join('/scratch/zliutkus', x)

def setup(size=250):
    with open(scratch_path('data/voc-classes.txt'), 'r') as file:
        classes = file.readlines()
    split = size//len(classes)
    used = set()
    for class_name in classes:
        class_name = class_name.strip()+'_train.txt'
        with open(scratch_path(os.path.join('data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/', class_name)), 'r') as file0:
            names = [x.split(' ')[0] for x in file0.readlines() if x.split(' ')[1].strip() != '-1']
        picked = random.sample(names, split)
        print(class_name, ": ",picked)
        for name in picked:
            name = name.split(' ')[0]
            if name in used:
                continue
            used.add(name)
            shutil.move(scratch_path('data/VOC/train/images/'+name+'.jpg'), scratch_path('data/VOC/semi-supervised/images/'+name+'.jpg'))
            shutil.copy(scratch_path('data/VOC/train/labels/'+name+'.txt'), scratch_path('data/VOC/semi-supervised/labels/'+name+'.txt'))


def reset():
    for name in os.listdir(scratch_path('data/VOC/semi-supervised/images/')):
        shutil.move(scratch_path('data/VOC/semi-supervised/images/'+name), scratch_path('data/VOC/train/images/'+name))
        os.remove(scratch_path('data/VOC/semi-supervised/labels/'+name.replace('.jpg', '.txt')))

def student_iteration(teacher):
    # Train student model
    print("New base weights: {}".format(os.path.join(teacher.trainer.save_dir, 'weights', 'best.pt')))
    student = YOLO(os.path.join(teacher.trainer.save_dir, 'weights', 'best.pt'))
    student.train(data='VOC.yaml', epochs=150, patience=25, device=0, pretrained=True, project=home_path('runs/train'), name='train') # Change save dir project and name, where save_dir=project/name
    return student


# Setup data subset for semi-supervised learning
reset()
setup(560)  # Approx. 10% of training data

# Train teacher model
teacher = YOLO('yolov8n.pt')
teacher.train(data='VOC.yaml', epochs=150, patience=25, device=0, project=home_path('runs/train'), name='train') # Change save dir project and name, where save_dir=project/name

# Iteratively assign pseudo-labels and train student model
n = 3
for _ in range(n):

    results = teacher.predict(source=scratch_path('data/VOC/train/images'), stream=True, conf=0.95)
    for result in results:
        if result.boxes.shape[0] > 0:
            result.save_txt(scratch_path('data/VOC/semi-supervised/labels/'+os.path.basename(result.path).replace('.jpg', '.txt')))
            shutil.move(scratch_path('data/VOC/train/images/'+os.path.basename(result.path)), scratch_path('data/VOC/semi-supervised/images/'+os.path.basename(result.path)))

    teacher = student_iteration(teacher)
