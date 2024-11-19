import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical

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

# Verileri yükle ve etiketleri oluştur
mild_demented_data, mild_demented_labels = load_and_preprocess_data('C:/Users/iyagm/Anaconda/Tez/Dataset/Mild_Demented', 0)
moderate_demented_data, moderate_demented_labels = load_and_preprocess_data('C:/Users/iyagm/Anaconda/Tez/Dataset/Moderate_Demented', 1)
non_demented_data, non_demented_labels = load_and_preprocess_data('C:/Users/iyagm/Anaconda/Tez/Dataset/Non_Demented', 2)
very_mild_demented_data, very_mild_demented_labels = load_and_preprocess_data('C:/Users/iyagm/Anaconda/Tez/Dataset/Very_Mild_Demented', 3)

# Verileri dört sınıfa göre ayır
mild_train_data, mild_test_data, mild_train_labels, mild_test_labels = train_test_split(mild_demented_data, mild_demented_labels, test_size=0.3, random_state=42)
moderate_train_data, moderate_test_data, moderate_train_labels, moderate_test_labels = train_test_split(moderate_demented_data, moderate_demented_labels, test_size=0.3, random_state=42)
non_train_data, non_test_data, non_train_labels, non_test_labels = train_test_split(non_demented_data, non_demented_labels, test_size=0.3, random_state=42)
very_mild_train_data, very_mild_test_data, very_mild_train_labels, very_mild_test_labels = train_test_split(very_mild_demented_data, very_mild_demented_labels, test_size=0.3, random_state=42)

# Train verilerini birleştir
train_data = np.concatenate((mild_train_data, moderate_train_data, non_train_data, very_mild_train_data), axis=0)
train_labels = np.concatenate((mild_train_labels, moderate_train_labels, non_train_labels, very_mild_train_labels), axis=0)

# Test verilerini birleştir
test_data = np.concatenate((mild_test_data, moderate_test_data, non_test_data, very_mild_test_data), axis=0)
test_labels = np.concatenate((mild_test_labels, moderate_test_labels, non_test_labels, very_mild_test_labels), axis=0)

# Etiketleri one-hot encoding'e çevir
train_labels_one_hot = to_categorical(train_labels, num_classes=4)
test_labels_one_hot = to_categorical(test_labels, num_classes=4)

# Verileri normalize et
train_data = train_data / 255.0
test_data = test_data / 255.0


'''
# K-NN modelini oluştur
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Modeli eğit
knn_model.fit(train_data.reshape(train_data.shape[0], -1), np.argmax(train_labels_one_hot, axis=1))

# 5 kat çapraz doğrulama için StratifiedKFold kullanarak skorları al
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(knn_model, train_data.reshape(train_data.shape[0], -1), np.argmax(train_labels_one_hot, axis=1), cv=cv, scoring='accuracy')


# Skorları ekrana yazdır
print("Accuracy Scores for Each Fold:", scores)

# Ortalama doğruluk skorunu yazdır
print("Mean Accuracy:", np.mean(scores))

# Test verileri üzerinde tahmin yap
labels_pred_flat = knn_model.predict(test_data.reshape(test_data.shape[0], -1))

# Karmaşıklık Matrisi
conf_matrix_knn = confusion_matrix(np.argmax(test_labels_one_hot, axis=1), labels_pred_flat)

# Karmaşıklık matrisini seaborn kullanarak görselleştir
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Karmaşıklık Matrisi (K-NN)')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

# Sınıflandırma Raporu
print(classification_report(np.argmax(test_labels_one_hot, axis=1), labels_pred_flat, target_names=class_names))

'''
# K-NN modelini oluştur
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Modeli eğit
knn_model.fit(train_data.reshape(train_data.shape[0], -1), np.argmax(train_labels_one_hot, axis=1))

# 5 kat çapraz doğrulama için StratifiedKFold kullanarak skorları al
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(knn_model, train_data.reshape(train_data.shape[0], -1), np.argmax(train_labels_one_hot, axis=1), cv=cv, scoring='accuracy')

# Skorları ekrana yazdır
print("Accuracy Scores for Each Fold:", scores)

# Ortalama doğruluk skorunu yazdır
print("Mean Accuracy:", np.mean(scores))

# Test verileri üzerinde tahmin yap
labels_pred_flat = knn_model.predict(test_data.reshape(test_data.shape[0], -1))

# Karmaşıklık Matrisi
conf_matrix_knn = confusion_matrix(np.argmax(test_labels_one_hot, axis=1), labels_pred_flat)

# Karmaşıklık matrisini seaborn kullanarak görselleştir
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Karmaşıklık Matrisi (K-NN)')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

# Sınıflandırma Raporu
print(classification_report(np.argmax(test_labels_one_hot, axis=1), labels_pred_flat, target_names=class_names))
