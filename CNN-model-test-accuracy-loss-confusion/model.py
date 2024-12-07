import logging
import os
from keras import Sequential, Input
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Loglama seviyesini ayarla
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Veri yolları
dataset_path = 'C:/Users/gulme/Downloads/fruit-dataset2/fruit-dataset'
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')

# Görüntü boyutu ve batch size
img_size = (224, 224)
batch_size = 32

# Veri artırımı ve normalizasyon
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Eğitim verisi generator
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# Test verisi generator
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# Model tanımı
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Modeli derle
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=25,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    verbose=1
)

# Modeli kaydet
model.save('fruit-model.h5')
print("Model saved!")

# Eğitim geçmişini kaydetmek için dön
import pickle
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
