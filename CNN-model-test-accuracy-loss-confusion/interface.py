import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import ImageTk, Image

# Load the model
model = load_model('D:/pycharm-project/pythonProject/fruit-model.h5')

# Define the class labels
class_labels = ['fresh_carrot', 'fresh_apple', 'fresh_banana', 'fresh_grape', 'fresh_orange',
                'rotten_apple', 'rotten_banana', 'rotten_carrot', 'rotten_grape', 'rotten_orange']

# Fruit properties
fruit_properties = {
    "apple": {"Calories": 52, "Protein": 0.3, "Fat": 0.2, "Carbohydrate": 14, "Recycle": "Shoe Sole Production"},
    "orange": {"Calories": 47, "Protein": 0.9, "Fat": 0.1, "Carbohydrate": 12, "Recycle": "Cosmetics Sector"},
    "grape": {"Calories": 66, "Protein": 0.6, "Fat": 0.4, "Carbohydrate": 17, "Recycle": "Bioplastic"},
    "banana": {"Calories": 88, "Protein": 1.1, "Fat": 0.3, "Carbohydrate": 23, "Recycle": "Fiber Production"},
    "carrot": {"Calories": 41, "Protein": 0.9, "Fat": 0.2, "Carbohydrate": 10, "Recycle": "Natural Food Coloring"},
}


# Function to predict the image
def predict_image(image_path):
    try:
        # Load the image and resize it to the model input size
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)  # Convert the image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch size
        img_array = img_array / 255.0  # Normalize the image

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class index
        predicted_class_label = class_labels[predicted_class]  # Get the label
        confidence = np.max(predictions) * 100  # Get confidence (probability)

        # Return prediction and confidence
        return predicted_class_label, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None


# Function to open a file dialog and select an image
def open_image():
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        display_image(file_path)
        predicted_class, confidence = predict_image(file_path)

        if predicted_class and confidence is not None:
            result_label.config(text=f"Predicted Class: {predicted_class}")
            confidence_label.config(text=f"Confidence: {confidence:.2f}%")

            fruit = predicted_class.split('_')[-1]  # Get the fruit name
            if 'fresh' in predicted_class:
                freshness_label.config(text="Fresh")
                fruit_info = fruit_properties.get(fruit, {})
                if fruit_info:
                    details = (
                        f"Calories: {fruit_info['Calories']} kcal\n"
                        f"Protein: {fruit_info['Protein']} g\n"
                        f"Fat: {fruit_info['Fat']} g\n"
                        f"Carbohydrate: {fruit_info['Carbohydrate']} g"
                    )
                else:
                    details = "No information available."
            else:  # Rotten
                freshness_label.config(text="Rotten")
                details = f"Recycle Type: {fruit_properties.get(fruit, {}).get('Recycle', 'No recycle info available.')}"

            details_label.config(text=details)
        else:
            result_label.config(text="Error during prediction!")


# Function to display the selected image
def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((250, 250))
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img


# Create the Tkinter window
root = tk.Tk()
root.title("Fruit Classifier")
root.geometry("600x700")
root.config(bg="#f5f5f5")

# Create GUI elements
img_label = tk.Label(root, bg="#f5f5f5")
img_label.grid(row=0, column=0, columnspan=2, pady=20)

open_button = tk.Button(root, text="Open Image", command=open_image, bg="#4CAF50", fg="white", font=("Arial", 14, "bold"))
open_button.grid(row=1, column=0, columnspan=2, pady=15)

result_label = tk.Label(root, text="Predicted Class: ", font=("Arial", 14), fg="blue", bg="#f5f5f5")
result_label.grid(row=2, column=0, columnspan=2, sticky="ew")

confidence_label = tk.Label(root, text="Confidence: ", font=("Arial", 14), fg="green", bg="#f5f5f5")
confidence_label.grid(row=3, column=0, columnspan=2, sticky="ew")

freshness_label = tk.Label(root, text="Fresh or Rotten: ", font=("Arial", 14), fg="red", bg="#f5f5f5")
freshness_label.grid(row=4, column=0, columnspan=2, sticky="ew")

details_label = tk.Label(root, text="", font=("Arial", 14), fg="black", bg="#f5f5f5", justify="left")
details_label.grid(row=5, column=0, columnspan=2, sticky="w", padx=10)

# Run the application
root.mainloop()
