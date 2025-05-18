import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import joblib
import os
import numpy as np

def setup_styles():
    style = ttk.Style()
    style.theme_use('clam')

    style.configure('TButton',
                    background='#1e1e1e',
                    foreground='white',
                    font=('Segoe UI', 11, 'bold'),
                    padding=8)
    style.map('TButton',
              background=[('active', '#2c2c2c')],
              foreground=[('active', 'white')])

    style.configure('TLabel',
                    background='#1e1e1e',
                    foreground='white',
                    font=('Segoe UI', 12))

    style.configure('TEntry',
                    fieldbackground='#2a2a2a',
                    foreground='white',
                    font=('Segoe UI', 12),
                    padding=5)

root = tk.Tk()
root.title("Customer Churn Prediction")
root.geometry("900x600")  # fenêtre plus grande
root.configure(bg="#1e1e1e")

setup_styles()

input_frame = tk.Frame(root, bg="#1e1e1e", padx=20, pady=20)
input_frame.pack(fill='x', padx=20, pady=10)

ttk.Label(input_frame, text="Durée du client (tenure) :").grid(row=0, column=0, sticky='w', pady=8)
tenure_entry = ttk.Entry(input_frame, width=30)
tenure_entry.grid(row=0, column=1, pady=8, padx=10)

ttk.Label(input_frame, text="Monthly Charges (0=mensuel,1=annuel,2=2ans) :").grid(row=1, column=0, sticky='w', pady=8)
monthly_entry = ttk.Entry(input_frame, width=30)
monthly_entry.grid(row=1, column=1, pady=8, padx=10)

ttk.Label(input_frame, text="Total Charges (frais) :").grid(row=2, column=0, sticky='w', pady=8)
total_entry = ttk.Entry(input_frame, width=30)
total_entry.grid(row=2, column=1, pady=8, padx=10)

model_linear_path = os.path.join("model", "regression_model.pkl")
model_forest_path = os.path.join("model", "randomForest_model.pkl")

try:
    model_linear = joblib.load(model_linear_path)
    model_forest = joblib.load(model_forest_path)
except Exception as e:
    messagebox.showerror("Erreur", f"Erreur lors du chargement des modèles : {e}")
    root.destroy()

def predict_linear():
    try:
        tenure = float(tenure_entry.get())
        monthly = float(monthly_entry.get())
        total = float(total_entry.get())
        prediction = model_linear.predict([[tenure, monthly, total]])
        pred_val = float(np.round(prediction[0], 4))
        result_linear.config(text=f"Régression (score continu) : {pred_val}")
    except Exception as e:
        messagebox.showerror("Erreur", str(e))

def predict_forest():
    try:
        tenure = float(tenure_entry.get())
        monthly = float(monthly_entry.get())
        total = float(total_entry.get())
        prediction = model_forest.predict([[tenure, monthly, total]])
        label = "Churn" if int(prediction[0]) == 1 else "Pas de churn"
        result_forest.config(text=f"Random Forest (classification) : {label}")
    except Exception as e:
        messagebox.showerror("Erreur", str(e))

def reset_fields():
    tenure_entry.delete(0, tk.END)
    monthly_entry.delete(0, tk.END)
    total_entry.delete(0, tk.END)
    result_linear.config(text="")
    result_forest.config(text="")

btn_frame = tk.Frame(root, bg="#1e1e1e")
btn_frame.pack(pady=20)

btn_linear = ttk.Button(btn_frame, text="Prédire avec Régression", command=predict_linear)
btn_linear.grid(row=0, column=0, padx=10, ipadx=10)

btn_forest = ttk.Button(btn_frame, text="Prédire avec Random Forest", command=predict_forest)
btn_forest.grid(row=0, column=1, padx=10, ipadx=10)

btn_reset = ttk.Button(btn_frame, text="Réinitialiser", command=reset_fields)
btn_reset.grid(row=0, column=2, padx=10, ipadx=10)

result_linear = tk.Label(root, text="", bg="#1e1e1e", fg="white", font=("Segoe UI", 14))
result_linear.pack(pady=(10, 5))

result_forest = tk.Label(root, text="", bg="#1e1e1e", fg="white", font=("Segoe UI", 14))
result_forest.pack(pady=(5, 20))

root.mainloop()
