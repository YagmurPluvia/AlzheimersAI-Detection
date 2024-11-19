import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score , f1_score, cohen_kappa_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

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

# Verileri birleştir
data = np.array(mild_demented_data + moderate_demented_data + non_demented_data + very_mild_demented_data)
labels = np.array(mild_demented_labels + moderate_demented_labels + non_demented_labels + very_mild_demented_labels)

# Etiketleri one-hot encoding'e çevir
labels_one_hot = to_categorical(labels, num_classes=4)

# Veriyi eğitim, doğrulama ve test setlerine ayır
train_data, test_data, train_labels, test_labels = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=42)

# Piksel değerlerini 0 ile 1 arasında normalize et
train_data = train_data / 255.0
val_data = val_data / 255.0
test_data = test_data / 255.0

# CNN modelini oluştur
input_layer = Input(shape=(64, 64, 1))
conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D((2, 2))(conv2)
flatten = Flatten()(pool2)
# CNN özellik çıkarma
cnn_features = Dense(128, activation='relu')(flatten)
cnn_features = Dense(4, activation='softmax')(cnn_features)

# CNN modelini ve özellik çıkarma katmanını birleştir
cnn_model = Model(inputs=input_layer, outputs=cnn_features)

# CNN modelini derleme
history = cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitim için ayarla
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# CNN modelini eğitme
cnn_model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels), callbacks=[early_stopping])

# CNN özellik çıkarma
cnn_features_train = cnn_model.predict(train_data)
cnn_features_test = cnn_model.predict(test_data)
cnn_features_val = cnn_model.predict(val_data)

# K-NN with GridSearchCV
param_grid = {'n_neighbors': [5], 'metric': ['euclidean']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3, cv=5)
grid.fit(cnn_features_train, np.argmax(train_labels, axis=1))

# En iyi parametreleri ve skoru alın
print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

# KNN sınıflandırıcı
KNN_classifier = grid.best_estimator_

# Test seti üzerinde tahmin yapma
KNN_pred_labels = KNN_classifier.predict(cnn_features_test)

# Performansı değerlendirme
accuracy = accuracy_score(np.argmax(test_labels, axis=1), KNN_pred_labels)
print("KNN Model Accuracy:", accuracy)


# Karmaşıklık Matrisi
conf_matrix_knn = confusion_matrix(np.argmax(test_labels, axis=1), KNN_pred_labels)

# Karmaşıklık matrisini seaborn kullanarak görselleştir
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (CNN-KNN)')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.show()

# Sınıflandırma Raporu
print(classification_report(np.argmax(test_labels, axis=1), KNN_pred_labels, target_names=class_names, zero_division=1,digits=6))

# Doğruluk
accuracy = accuracy_score(np.argmax(test_labels, axis=1), KNN_pred_labels)
print("Doğruluk:", accuracy)

# F1 skoru
f1 = f1_score(np.argmax(test_labels, axis=1), KNN_pred_labels, average='weighted')
print("F1 Skoru:", f1)

# Kappa katsayısı
kappa = cohen_kappa_score(np.argmax(test_labels, axis=1), KNN_pred_labels)
print("Kappa Katsayısı:", kappa)
