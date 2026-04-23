import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. SMART PATH FINDER ---
possible_paths = ['./', './chest_xray/', '../chest_xray/']
base_path = None

print("Searching for dataset folders...")
for p in possible_paths:
    if os.path.exists(os.path.join(p, 'train')):
        base_path = p
        print(f"✅ Found data in: {os.path.abspath(p)}")
        break

if base_path is None:
    print("❌ ERROR: Could not find 'train' folder!")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Files here: {os.listdir('.')}")
    exit()

if not os.path.exists('output'):
    os.makedirs('output')

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

# --- 2. DATA LOADING FUNCTIONS ---
def get_data(data_dir):
    data = []
    path_to_load = os.path.join(base_path, data_dir)
    print(f"Loading {data_dir} from {path_to_load}...")
    for label in labels:
        path = os.path.join(path_to_load, label)
        class_num = labels.index(label)
        if not os.path.exists(path): continue
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) 
                data.append([resized_arr, class_num])
            except: continue
    return data

def preprocess(data):
    if not data: return np.array([]), np.array([])
    x, y = [], []
    for f, l in data:
        x.append(f); y.append(l)
    return np.array(x).reshape(-1, img_size, img_size, 1) / 255.0, np.array(y)

# Load Data
x_train, y_train = preprocess(get_data('train'))
x_test, y_test = preprocess(get_data('test'))
x_val, y_val = preprocess(get_data('val'))

# --- 3. ANALYSIS 1: BAR GRAPH ---
plt.figure(figsize=(8,6))
counts = [len(y_train[y_train==0]), len(y_train[y_train==1])]
sns.barplot(x=labels, y=counts, hue=labels, palette="viridis", legend=False)
plt.title('Data Distribution')
plt.savefig('output/1_bar_graph.png')
plt.close()

# --- 4. MODEL & TRAINING ---
model = models.Sequential([
    layers.Input(shape=(150, 150, 1)),
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.1, horizontal_flip=True)
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.3, min_lr=0.000001)

print("\nStarting Training...")
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), 
                    epochs=10, validation_data=(x_val, y_val), callbacks=[lr_reduction])

# --- 5. GENERATE ALL REQUESTED OUTPUTS ---
print("\nGenerating Analysis...")

# Acc/Loss Curves
plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy'); plt.legend(); plt.savefig('output/2_accuracy.png'); plt.close()

plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss'); plt.legend(); plt.savefig('output/3_loss.png'); plt.close()

# Confusion Matrix
preds_prob = model.predict(x_test)
preds_bin = (preds_prob > 0.5).astype("int32").flatten()
cm = confusion_matrix(y_test, preds_bin)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.savefig('output/4_matrix.png'); plt.close()

# Pie Chart
correct = np.sum(preds_bin == y_test)
incorrect = len(y_test) - correct
plt.pie([correct, incorrect], labels=['Correct', 'Incorrect'], autopct='%1.1f%%')
plt.savefig('output/5_pie_chart.png'); plt.close()

# Correct/Incorrect Samples
def save_samples(indices, name, title):
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices[:8]):
        plt.subplot(2, 4, i+1)
        plt.imshow(x_test[idx].reshape(150,150), cmap='gray')
        plt.title(f"P:{labels[preds_bin[idx]]}\nA:{labels[y_test[idx]]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.savefig(f'output/{name}.png'); plt.close()

save_samples(np.where(preds_bin == y_test)[0], '6_correct', 'Correct Samples')
save_samples(np.where(preds_bin != y_test)[0], '7_incorrect', 'Incorrect Samples')

print("\nDONE! All results are in the 'output' folder.")

model.save('pneumonia_model.h5')
print("Model saved successfully as pneumonia_model.h5")