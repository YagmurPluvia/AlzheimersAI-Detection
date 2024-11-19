import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pathlib 
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
# Karmaşıklık Matrisi (Confusion Matrix) Oluşturma
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

import time

# Başlangıç zamanını kaydet
start_time = time.time()

# GPU kullanılabilirliğini kontrol et
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU active! -", physical_devices)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("GPU not active!")

# Veri alma
path = 'C:/Users/iyagm/Anaconda/Tez/Dataset'
veri_küt = pathlib.Path(path)

# Veri etiket isimleri
sınıf_isim = np.array([sorted(item.name for item in veri_küt.glob("*"))])
print(sınıf_isim)

batch_size = 32
img_height = 180
img_width = 180
seed = 42

# Veriyi yükleyerek eğitim ve doğrulama setlerini oluşturma
train_data = image_dataset_from_directory(
    veri_küt,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_data = image_dataset_from_directory(
    veri_küt,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Model oluşturma
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    #layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(sınıf_isim[0]), activation="softmax")  # Sınıf sayısı kadar nöron
])

model.summary()

# Modeli derleme
model.compile(optimizer="Adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

# Modeli eğitme
epochs = 5  # Dilediğiniz sayıda epoch'u belirleyebilirsiniz
history = model.fit(train_data,
                    epochs=epochs,
                    validation_data=val_data, 
                    batch_size=batch_size)


# Eğitim sürecinden elde edilen accuracy ve loss değerleri
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Accuracy grafiği
plt.plot(range(1, epochs+1), train_accuracy, label='Train Accuracy')
plt.plot(range(1, epochs+1), val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss grafiği
plt.plot(range(1, epochs+1), train_loss, label='Train Loss')
plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Eğitim seti üzerinde tahminler yapma
train_predictions = model.predict(train_data)
train_pred_labels = np.argmax(train_predictions, axis=1)  # En yüksek olasılığa sahip sınıf etiketini al

# Doğrulama seti üzerinde tahminler yapma
val_predictions = model.predict(val_data)
val_pred_labels = np.argmax(val_predictions, axis=1)  # En yüksek olasılığa sahip sınıf etiketini al

# Eğitim seti etiketlerini alma
train_true_labels = np.concatenate([y for x, y in train_data], axis=0)

# Doğrulama seti etiketlerini alma
val_true_labels = np.concatenate([y for x, y in val_data], axis=0)

# Eğitim seti için karmaşıklık matrisi
train_cm = confusion_matrix(train_true_labels, train_pred_labels)

# Doğrulama seti için karmaşıklık matrisi
val_cm = confusion_matrix(val_true_labels, val_pred_labels)

# Eğitim seti karmaşıklık matrisini görselleştirme
plt.figure(figsize=(10, 6))
sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", xticklabels=sınıf_isim[0], yticklabels=sınıf_isim[0])
plt.title('Eğitim Seti Karmaşıklık Matrisi')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show()

#^^^^^^
# Doğrulama seti karmaşıklık matrisini görselleştirme
plt.figure(figsize=(10, 6))
sns.heatmap(val_cm, annot=True, fmt="d", cmap="Blues", xticklabels=sınıf_isim[0], yticklabels=sınıf_isim[0])
plt.title('Doğrulama Seti Karmaşıklık Matrisi')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show()

from sklearn.metrics import classification_report

# Doğrulama seti için sınıf isimlerini al
class_names = sınıf_isim[0]
# Classification report'u elde etme
classification_rep = classification_report(val_true_labels, val_pred_labels, target_names=class_names)
# Yazdırma
print(classification_rep)


# Bitiş zamanını kaydet
end_time = time.time()

# İşlem süresini hesapla
process_time = end_time - start_time
print("Veri yükleme süresi:", process_time, "saniye")


