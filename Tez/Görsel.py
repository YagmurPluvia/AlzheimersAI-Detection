import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel_filter(img):
    # Sobel filtresi uygula
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    filtered_img = np.sqrt(sobelx**2 + sobely**2)
    return filtered_img

# Resimleri yükle ve ön işleme fonksiyonu
def load_and_preprocess_data(folder_path, label):
    original_data = []
    denoised_data = []
    sobel_data = []
    labels = []
    label_names = []
    for filename in os.listdir(folder_path)[:1]:  # Sadece ilk 4 fotoğrafı al
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        
        # Orjinal fotoğrafı ekle
        original_data.append(img)
        
        # Bilateral filtresi uygula (Gürültü azaltma)
        denoised_img = cv2.bilateralFilter(img, 9, 75, 75)
        denoised_data.append(denoised_img)
        
        # Sobel filtresi uygula
        sobel_img = apply_sobel_filter(denoised_img)
        sobel_data.append(sobel_img)
        
        labels.extend([label, label, label])  # Her fotoğraf için etiketi ekle
        label_names.extend([filename, filename, filename])  # Her fotoğraf için etiket adını ekle
        
    return original_data, denoised_data, sobel_data, labels, label_names

# Visualize images
def visualize_samples(original_data, denoised_data, sobel_data, label_names):
    plt.figure(figsize=(12, 8))
    num_samples = len(original_data)
    for i in range(num_samples):
        plt.subplot(3, num_samples, i+1)
        plt.imshow(original_data[i], cmap='gray')
        plt.title('Original\n' + label_names[i])
        plt.axis('off')
        
        plt.subplot(3, num_samples, num_samples+i+1)
        plt.imshow(denoised_data[i], cmap='gray')
        plt.title('Denoised\n' + label_names[i])
        plt.axis('off')
        
        plt.subplot(3, num_samples, 2*num_samples+i+1)
        plt.imshow(sobel_data[i], cmap='gray')
        plt.title('Sobel Filtered\n' + label_names[i])
        plt.axis('off')
    
    plt.subplots_adjust(hspace=0.5)  # Dikey boşluk ayarı
    plt.show()

# Define label names
label_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Load and preprocess data for each label category
original_data = []
denoised_data = []
sobel_data = []
labels = []
label_names_all = []

# Load and preprocess data for each label category
mild_demented_original, mild_demented_denoised, mild_demented_sobel, mild_demented_labels, mild_demented_label_names = load_and_preprocess_data('C:/Users/iyagm/Anaconda/Tez/Dataset/Mild_Demented', 0)
moderate_demented_original, moderate_demented_denoised, moderate_demented_sobel, moderate_demented_labels, moderate_demented_label_names = load_and_preprocess_data('C:/Users/iyagm/Anaconda/Tez/Dataset/Moderate_Demented', 1)
non_demented_original, non_demented_denoised, non_demented_sobel, non_demented_labels, non_demented_label_names = load_and_preprocess_data('C:/Users/iyagm/Anaconda/Tez/Dataset/Non_Demented', 2)
very_mild_demented_original, very_mild_demented_denoised, very_mild_demented_sobel, very_mild_demented_labels, very_mild_demented_label_names = load_and_preprocess_data('C:/Users/iyagm/Anaconda/Tez/Dataset/Very_Mild_Demented', 3)

# Combine data and labels
original_data = mild_demented_original + moderate_demented_original + non_demented_original + very_mild_demented_original
denoised_data = mild_demented_denoised + moderate_demented_denoised + non_demented_denoised + very_mild_demented_denoised
sobel_data = mild_demented_sobel + moderate_demented_sobel + non_demented_sobel + very_mild_demented_sobel
labels = mild_demented_labels + moderate_demented_labels + non_demented_labels + very_mild_demented_labels
label_names_all = mild_demented_label_names + moderate_demented_label_names + non_demented_label_names + very_mild_demented_label_names

# Visualize images
visualize_samples(original_data, denoised_data, sobel_data, label_names_all)
