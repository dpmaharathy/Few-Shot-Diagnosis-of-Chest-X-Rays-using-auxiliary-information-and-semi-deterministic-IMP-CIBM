import pandas as pd
import os
import cv2
import numpy as np

# List of possible labels
labels = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
    'Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other',
    'Pneumonia','Pneumothorax','Support Devices'
]

# df = pd.read_csv('/DATA2/MIMIC-CXR/mimic-cxr-2.0.0-chexpert.csv')
# df = df.f
# Define directories
output_dir = '/DATA1/MIMIC_CXR_classwise_3k'
numpy_mimic_path = '/home/maharathy1/mimic_128/'

# Lists of labels for splits
test_l=[['Atelectasis','Cardiomegaly','Consolidation'],
        ['Edema','Enlarged Cardiomediastinum','Fracture'],
        ['Lung Lesion','Lung Opacity','No Finding'],
        ['Pleural Effusion','Pleural Other', 'Pneumonia'],
        ['Pneumothorax','Support Devices','Atelectasis']]

val_l=[['Edema','Enlarged Cardiomediastinum','Fracture'],
['Lung Lesion','Lung Opacity','No Finding'],
['Pleural Effusion','Pleural Other', 'Pneumonia'],
['Pneumothorax','Support Devices','Atelectasis'],
['Cardiomegaly','Consolidation','Edema']]

train_l=[['Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other',
    'Pneumonia','Pneumothorax','Support Devices'],
['Atelectasis','Cardiomegaly','Consolidation','Pleural Effusion','Pleural Other',
    'Pneumonia','Pneumothorax','Support Devices'],
['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
    'Fracture','Pneumothorax','Support Devices'],
['Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
    'Fracture','Lung Lesion','Lung Opacity','No Finding'],
['Enlarged Cardiomediastinum',
    'Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other',
    'Pneumonia']]

def read_images_from_dir(dir_path, label):
    X = []
    y = []
    for file in os.listdir(dir_path):
        if file.endswith('.jpg'):
            img_path = os.path.join(dir_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            X.append(img/255)
            # print(label)
            y.append(label)
            # print(labels.index(label))
    return X, y

for i in range(5):
    print(f"Processing split {i+1}...")
    
    # Initialize lists to hold images and labels for each split
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    # Read images for test split
    for label in test_l[i]:
        dir_path = os.path.join(output_dir, label.replace(" ", "_"))
        X_test_temp, y_test_temp = read_images_from_dir(dir_path, label)
        X_test.extend(X_test_temp)
        y_test.extend(y_test_temp)
    
    # Read images for validation split
    for label in val_l[i]:
        dir_path = os.path.join(output_dir, label.replace(" ", "_"))
        X_val_temp, y_val_temp = read_images_from_dir(dir_path, label)
        X_val.extend(X_val_temp)
        y_val.extend(y_val_temp)
    
    # Read images for training split
    for label in train_l[i]:
        dir_path = os.path.join(output_dir, label.replace(" ", "_"))
        X_train_temp, y_train_temp = read_images_from_dir(dir_path, label)
        X_train.extend(X_train_temp)
        y_train.extend(y_train_temp)
    
    
    # Convert to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Save numpy arrays
    np.save(f'{numpy_mimic_path}Xtrain_{i+1}.npy', X_train)
    np.save(f'{numpy_mimic_path}ytrain_{i+1}.npy', y_train)
    np.save(f'{numpy_mimic_path}Xval_{i+1}.npy', X_val)
    np.save(f'{numpy_mimic_path}yval_{i+1}.npy', y_val)
    np.save(f'{numpy_mimic_path}Xtest_{i+1}.npy', X_test)
    np.save(f'{numpy_mimic_path}ytest_{i+1}.npy', y_test)
    
    # Print unique labels and counts
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))
    print(np.unique(y_test, return_counts=True))
