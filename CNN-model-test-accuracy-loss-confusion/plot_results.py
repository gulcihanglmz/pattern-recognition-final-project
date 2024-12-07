import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load training history
with open('history.pkl', 'rb') as f:
    history = pickle.load(f)

# Loss Graph
plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label='Training Loss')
if 'val_loss' in history:
    plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss vs Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Accuracy Graph
plt.figure(figsize=(12, 6))
plt.plot(history['accuracy'], label='Training Accuracy')
if 'val_accuracy' in history:
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy vs Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Load model
model = load_model('fruit-model.h5')  # Correct file name

# Create test dataset
test_dir = 'C:/Users/gulme/Downloads/fruit-dataset2/fruit-dataset/test'
img_size = (224, 224)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # It should be ordered
)

# Get predictions
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = [k for k, v in sorted(test_generator.class_indices.items(), key=lambda item: item[1])]

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.show()
