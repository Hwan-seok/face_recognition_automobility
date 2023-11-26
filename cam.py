import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow import keras
import numpy as np

# Load your trained model
model = keras.models.load_model('face_recognition_automobility/saved_model/main.h5')

def update_image():
    ret, frame = cap.read()
    if ret:
        # Resize and preprocess the frame
        resized_frame = cv2.resize(frame, (224, 224))
        processed_frame = resized_frame / 255.0
        batched_frame = processed_frame.reshape(1, 224, 224, 3)

        # Predict
        predictions = model.predict(batched_frame)
        predicted_classes = np.argmax(predictions, axis=1)

        # You can process the predictions here
        # ...
        print(predictions)
        print(predicted_classes)

        # Convert to a format Tkinter can use
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        label_img.imgtk = imgtk
        label_img.configure(image=imgtk)

        # Update predictions label
        label_pred.config(text=str(predictions))  # Update with actual prediction result

    label_img.after(20, update_image)  # Update image every 20 ms

# Initialize Tkinter window
root = tk.Tk()
root.title("Camera Feed with Predictions")

# Create a label in the window to hold the camera frames
label_img = tk.Label(root)
label_img.pack()

# Label for predictions
label_pred = tk.Label(root, text="Predictions")
label_pred.pack()

# Set up camera capture
cap = cv2.VideoCapture(0)

# Start the update process
update_image()

# Start the GUI loop
root.mainloop()

# Release the camera when the window is closed
cap.release()
