import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resimleri yükle ve ön işleme fonksiyonu
def load_and_preprocess_data(folder_path, label):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        data.append(img)
        labels.append(label)
    return data, labels

# Visualize one image from each label category vertically
def visualize_samples(data, labels, label_names):
    plt.figure(figsize=(8, 12))
    for i in range(len(label_names)):
        img_index = labels.index(i)
        plt.subplot(len(label_names), 1, i+1)
        plt.imshow(data[img_index], cmap='gray')
        plt.title(label_names[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Define label names
label_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Load and preprocess data
mild_demented_data, mild_demented_labels = load_and_preprocess_data('C:/Users/iyagm/Anaconda/Tez/Dataset/Mild_Demented', 0)
moderate_demented_data, moderate_demented_labels = load_and_preprocess_data('C:/Users/iyagm/Anaconda/Tez/Dataset/Moderate_Demented', 1)
non_demented_data, non_demented_labels = load_and_preprocess_data('C:/Users/iyagm/Anaconda/Tez/Dataset/Non_Demented', 2)
very_mild_demented_data, very_mild_demented_labels = load_and_preprocess_data('C:/Users/iyagm/Anaconda/Tez/Dataset/Very_Mild_Demented', 3)

# Combine data and labels
all_data = mild_demented_data + moderate_demented_data + non_demented_data + very_mild_demented_data
all_labels = mild_demented_labels + moderate_demented_labels + non_demented_labels + very_mild_demented_labels

# Visualize one sample from each label category vertically
visualize_samples(all_data, all_labels, label_names)
