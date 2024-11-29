import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import ImageTk, Image

# Load the model
model = load_model('D:/pycharm-project/pythonProject/fruit_model.h5')

# Define the class labels
class_labels = ['fresh_carrot', 'fresh_apple', 'fresh_banana', 'fresh_grape', 'fresh_orange',
                'rotten_apple', 'rotten_banana', 'rotten_carrot', 'rotten_grape', 'rotten_orange']


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
    # Open a file dialog to choose an image file
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        # Display the selected image in the Tkinter window
        display_image(file_path)

        # Get the prediction
        predicted_class, confidence = predict_image(file_path)

        # Update the result labels with prediction and confidence
        if predicted_class and confidence is not None:
            result_label.config(text=f"Predicted Class: {predicted_class}")
            confidence_label.config(text=f"Confidence: {confidence:.2f}%")
            freshness_label.config(text=f"Fresh or Rotten: {'Fresh' if 'fresh' in predicted_class else 'Rotten'}")
        else:
            result_label.config(text="Error during prediction!")


# Function to display the selected image in the Tkinter window
def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((250, 250))  # Resize image to fit the window
    img = ImageTk.PhotoImage(img)

    # Update the image on the label
    img_label.config(image=img)
    img_label.image = img


# Create the Tkinter window
root = tk.Tk()
root.title("Fruit Classifier")
root.geometry("600x600")  # Set fixed window size

# Add background color and font styles
root.config(bg="#f5f5f5")  # Light gray background

# Create a label to display the image
img_label = tk.Label(root, bg="#f5f5f5")  # Set background color to match the window
img_label.grid(row=0, column=0, columnspan=2, pady=20)

# Create a button to open the image with a custom style
open_button = tk.Button(root, text="Open Image", command=open_image, bg="#4CAF50", fg="white", font=("Arial", 14, "bold"), relief="raised", bd=2)
open_button.grid(row=1, column=0, columnspan=2, pady=15)

# Create labels for the prediction results using grid layout
result_label = tk.Label(root, text="Predicted Class: ", font=("Helvetica", 16, "italic"), fg="#2c3e50", bg="#f5f5f5")
result_label.grid(row=2, column=0, pady=5, columnspan=2, sticky="ew")

confidence_label = tk.Label(root, text="Confidence: ", font=("Helvetica", 16, "italic"), fg="#2c3e50", bg="#f5f5f5")
confidence_label.grid(row=3, column=0, pady=5, columnspan=2, sticky="ew")

freshness_label = tk.Label(root, text="Fresh or Rotten: ", font=("Helvetica", 16, "italic"), fg="#2c3e50", bg="#f5f5f5")
freshness_label.grid(row=4, column=0, pady=5, columnspan=2, sticky="ew")

# Run the Tkinter event loop
root.mainloop()
