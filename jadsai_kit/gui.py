# jadsai_kit/gui.py
import tkinter as tk
from tkinter import messagebox
from .model_builder import create_cnn_model

class JadsAIKitGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("JadsAI_Kit")
        self.root.geometry("400x300")
        
        tk.Label(self.root, text="JadsAI_Kit: Neural Network Builder", font=("Arial", 14)).pack(pady=10)
        tk.Button(self.root, text="Create CNN Model", command=self.create_model).pack(pady=10)
        self.status_label = tk.Label(self.root, text="Status: Ready", font=("Arial", 10))
        self.status_label.pack(pady=10)
    
    def create_model(self):
        try:
            model = create_cnn_model()
            self.status_label.config(text="Status: CNN Model Created Successfully!")
            messagebox.showinfo("Success", "CNN Model created with TensorFlow!")
        except Exception as e:
            self.status_label.config(text=f"Status: Error - {str(e)}")
            messagebox.showerror("Error", f"Failed to create model: {str(e)}")
    
    def run(self):
        self.root.mainloop()
