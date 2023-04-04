import cv2
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
cap = cv2.VideoCapture(file_path)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
