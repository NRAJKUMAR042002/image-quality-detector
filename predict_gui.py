import tensorflow as tf
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

model = tf.keras.models.load_model("model.h5")

def predict_image():
    file_path = filedialog.askopenfilename()

    img = Image.open(file_path)
    img = img.resize((128,128))

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = "GOOD IMAGE"
    else:
        result = "BAD IMAGE"

    label_result.config(text=result)

root = Tk()
root.title("Image Quality Detector")
root.geometry("400x300")

btn = Button(root,text="Upload Image",command=predict_image)
btn.pack(pady=20)

label_result = Label(root,text="Result will appear here",font=("Arial",16))
label_result.pack(pady=20)

root.mainloop()