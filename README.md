## Effects of adding unlabeled data through pseudo-labeling

This repository contains the code for the experiments in the bachelor thesis "Effects of adding unlabeled data through pseudo-labeling" by Žygimantas Liutkus.
Experiments for the thesis were conducted using the Ultralytics YOLOv8n model and evaluated on the PASCAL VOC2012 dataset. The code is written in Python and uses the PyTorch library.
To ensure reproducibility, comments are added to the code to explain the purpose of each part of the code. Instructions on how to run the code are provided below.

### Structure
The repository is structured as follows:
- `data` - directory for the PASCAL VOC2012 dataset (or other future datasets, in YOLO format).
- `logs` - directory for the logs of the experiments conducted for the thesis.
  - `added_object_counts` - logs of the number of objects added to the training set per iteration in the thesis experiments.
  - `colab_results` - logs of the experiments conducted on Google Colab.
  - `delftblue_results` - logs of the experiments conducted on the DelftBlue cluster.
  - `torchmetrics_map` - logs of the mAP scores of the models recomputed using the *torchmetrics* library.
- `trained_models` - directory containing the models trained during the thesis experiments.
- `convert.py` - script to convert the PASCAL VOC2012 dataset to the YOLO format and setup the directories.
- `label.py` - a python file containing lists of labels from the PASCAL VOC2012 dataset used in the naive teacher-student experiments. Acts as a randomisation seed, since labels are randomly selected during the setup step of the teacher-student script.
- `main.py` - the main script file, containing code for plotting and calculating results.
- `teacher_student.py` - script used to conduct all experiments.
- `teacher-student-script.sh` - script used to run the teacher-student script on the DelftBlue cluster.
- `VOC.yaml` - configuration file for the PASCAL VOC2012 dataset.
- `VOC1.yaml` - configuration file for the PASCAL VOC2012 used for one of the ensemble models.
- `VOC2.yaml` - configuration file for the PASCAL VOC2012 used for the other ensemble model.
- `requirements.txt` - file containing the required packages for the project.
- `confidence_scaling` - file containing the modified non_max_suppression function for the Ultralytics library.

Some abreviations used in the project:
- `ts` - naive teacher-student models.
- `dt` - models trained with dynamic thresholding.
- `cs` - models trained with confidence scaling.
- `cs_dt` - models trained with both confidence scaling and dynamic thresholding.
- `ensemble` - ensemble models.
- files ending with `_10`, `_20` or `_50` - models trained with 10%, 20% or 50% of the VOC2012 training set as labeled data. The rest of the training data was used as unlabeled data.

### Setup
The code is written in Python 3.9 and requires the following setup steps:
1. Clone the repository.
2. Install the required packages by running `pip install -r requirements.txt`.
3. Download the PASCAL VOC2012 dataset from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar and move it to the `data` directory.
4. Execute the `convert.py` script to convert the dataset to the required YOLO format and setup the directories.

### Running the code
To run the experiments, first setup the variables within the `teacher_student.py` script. 
Code comments explain what each variable is used for.
To enable confidence scaling, the non_max_suppression function in the Ultralytics module needs to be modified. The modified function is provided in `confidence_scaling`. To use it, replace the non_max_suppression function in the `utils` module of the Ultralytics library with the provided functions.
Then execute the script by running `python teacher_student.py` in the console or by running the file in an IDE of your choice.