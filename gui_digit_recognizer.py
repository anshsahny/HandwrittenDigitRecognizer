import tensorflow as tf
import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import *

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

model = tf.keras.models.load_model('mnist.h5')


def testing():
    img = cv2.imread('image.png', 0)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    prediction = model.predict(img)
    return prediction


def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=40)
    draw.line([x1, y1, x2, y2], fill="black", width=40)


def model1():
    filename = "image.png"
    image1.save(filename)
    pred = testing()
    print('argmax', np.argmax(pred[0]), '\n',
          pred[0][np.argmax(pred[0])], '\n', classes[np.argmax(pred[0])])
    txt.insert(tk.INSERT,
               "The number is {}\nAccuracy: {}%".format(classes[np.argmax(pred[0])], round(pred[0][np.argmax(pred[0])] * 100, 2)))


def clear():
    cv.delete('all')
    draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
    txt.delete('1.0', END)


root = Tk()

root.resizable(0, 0)
cv = Canvas(root, width=500, height=500, bg='white')
cv.pack()

image1 = PIL.Image.new("RGB", (500, 500), (255, 255, 255))
draw = ImageDraw.Draw(image1)

txt = tk.Text(root, bd=3, exportselection=0, bg='WHITE', font='Helvetica',
              padx=10, pady=10, height=5, width=20)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

btnModel = Button(text="Predict the Digit", command=model1)
btnClear = Button(text="Clear Display", command=clear)
btnModel.pack()
btnClear.pack()
txt.pack()
root.title('GUI Digit Recognizer')
root.mainloop()
