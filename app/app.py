import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Style
def setup_styles():
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TButton', background='#1e1e1e', foreground='white', font=('Segoe UI', 11, 'bold'), padding=8)
    style.map('TButton', background=[('active', '#2c2c2c')], foreground=[('active', 'white')])
    style.configure('TLabel', background='#1e1e1e', foreground='white', font=('Segoe UI', 12))
    style.configure('TEntry', fieldbackground='#2a2a2a', foreground='white', font=('Segoe UI', 12), padding=5)

# Fenêtre principale
root = tk.Tk()
root.title("Customer Churn Prediction")
root.geometry("1100x850")
root.configure(bg="#1e1e1e")
setup_styles()

# Cadre des entrées
input_frame = tk.Frame(root, bg="#1e1e1e", padx=20, pady=20)
input_frame.pack(fill='x', padx=20, pady=10)

ttk.Label(input_frame, text="Durée du client (tenure) :").grid(row=0, column=0, sticky='w', pady=8)
tenure_entry = ttk.Entry(input_frame, width=30)
tenure_entry.grid(row=0, column=1, pady=8, padx=10)

ttk.Label(input_frame, text="Monthly Charges (0:monthly, 1:year, 2:two year) :").grid(row=1, column=0, sticky='w', pady=8)
monthly_entry = ttk.Entry(input_frame, width=30)
monthly_entry.grid(row=1, column=1, pady=8, padx=10)

ttk.Label(input_frame, text="Total Charges :").grid(row=2, column=0, sticky='w', pady=8)
total_entry = ttk.Entry(input_frame, width=30)
total_entry.grid(row=2, column=1, pady=8, padx=10)

# Chargement des modèles
model_linear_path = os.path.join("model", "regression_model.pkl")
model_forest_path = os.path.join("model", "randomForest_model.pkl")

try:
    model_linear = joblib.load(model_linear_path)
    model_forest = joblib.load(model_forest_path)
except Exception as e:
    messagebox.showerror("Erreur", f"Erreur lors du chargement des modèles : {e}")
    root.destroy()

# Fonctions de prédiction
def predict_linear():
    try:
        tenure = float(tenure_entry.get())
        monthly = float(monthly_entry.get())
        total = float(total_entry.get())
        prediction = model_linear.predict([[tenure, monthly, total]])
        pred_val = float(np.round(prediction[0], 4))
        result_linear.config(text=f"Probabilité de churn (score continu) : {pred_val}")
    except Exception as e:
        messagebox.showerror("Erreur", str(e))

def predict_forest():
    try:
        tenure = float(tenure_entry.get())
        monthly = float(monthly_entry.get())
        total = float(total_entry.get())
        prediction = model_forest.predict([[tenure, monthly, total]])
        label = "Churn" if int(prediction[0]) == 1 else "Pas de churn"
        result_forest.config(text=f"Prédiction du churn (classification): {label}" )
    except Exception as e:
        messagebox.showerror("Erreur", str(e))

# Reset
def reset_fields():
    tenure_entry.delete(0, tk.END)
    monthly_entry.delete(0, tk.END)
    total_entry.delete(0, tk.END)
    result_linear.config(text="")
    result_forest.config(text="")
    for widget in chart_frame_linear.winfo_children():
        widget.destroy()
    for widget in chart_frame_forest.winfo_children():
        widget.destroy()

# Afficher diagramme régression
def afficher_diagramme_regression():
    try:
        for widget in chart_frame_linear.winfo_children():
            widget.destroy()

        tenure_range = np.linspace(0, 72, 100)
        monthly_categories = [0, 1, 2]
        total_fixed = 2000  # tu peux adapter selon ton dataset

        colors = ['cyan', 'lime', 'magenta']

        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e1e1e')

        for i, monthly in enumerate(monthly_categories):
            X = np.column_stack((tenure_range, [monthly]*len(tenure_range), [total_fixed]*len(tenure_range)))
            y_pred = model_linear.predict(X)
            ax.plot(tenure_range, y_pred, label=f'Monthly Charges = {monthly}', color=colors[i])

        ax.set_title("Régression : Prédiction vs Tenure", color='white')
        ax.set_xlabel("Tenure (mois)", color='white')
        ax.set_ylabel("Score prédictif", color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame_linear)
        canvas.draw()
        canvas.get_tk_widget().pack()

    except Exception as e:
        messagebox.showerror("Erreur", str(e))


# Afficher diagramme Random Forest
# Afficher diagramme Random Forest
def afficher_diagramme_forest():
    try:
        for widget in chart_frame_forest.winfo_children():
            widget.destroy()
        monthly = float(monthly_entry.get())
        total = float(total_entry.get())

        tenures = np.linspace(0, 72, 100)
        X = np.column_stack((tenures, [monthly]*len(tenures), [total]*len(tenures)))

        # Vérifie si le modèle supporte predict_proba
        if hasattr(model_forest, 'predict_proba'):
            proba = model_forest.predict_proba(X)[:, 1]
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e1e1e')
            ax.plot(tenures, proba, color='orange', linewidth=2)
            ax.set_title("Random Forest : Probabilité de churn vs Tenure", color='white')
            ax.set_ylabel("Probabilité de churn", color='white')
        else:
            preds = model_forest.predict(X)
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e1e1e')
            ax.plot(tenures, preds, color='orange', linewidth=2)
            ax.set_title("Random Forest : Churn (0/1) vs Tenure", color='white')
            ax.set_ylabel("Churn (0/1)", color='white')

        ax.set_xlabel("Tenure (mois)", color='white')
        ax.tick_params(colors='white')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame_forest)
        canvas.draw()
        canvas.get_tk_widget().pack()
    except Exception as e:
        messagebox.showerror("Erreur", str(e))

    try:
        for widget in chart_frame_forest.winfo_children():
            widget.destroy()

        tenure_range = np.linspace(0, 72, 100)
        monthly_categories = [0, 1, 2]
        total_fixed = 2000  # idem, valeur fixe

        colors = ['cyan', 'lime', 'magenta']

        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e1e1e')

        for i, monthly in enumerate(monthly_categories):
            X = np.column_stack((tenure_range, [monthly]*len(tenure_range), [total_fixed]*len(tenure_range)))
            y_pred_proba = model_forest.predict_proba(X)[:, 1]  # probabilité de churn = 1
            ax.plot(tenure_range, y_pred_proba, label=f'Monthly Charges = {monthly}', color=colors[i])

        ax.set_title("Random Forest : Probabilité de churn vs Tenure", color='white')
        ax.set_xlabel("Tenure (mois)", color='white')
        ax.set_ylabel("Probabilité de churn", color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame_forest)
        canvas.draw()
        canvas.get_tk_widget().pack()

    except Exception as e:
        messagebox.showerror("Erreur", str(e))

# Boutons
btn_frame = tk.Frame(root, bg="#1e1e1e")
btn_frame.pack(pady=20)

ttk.Button(btn_frame, text="Prédire avec Régression", command=predict_linear).grid(row=0, column=0, padx=10)
ttk.Button(btn_frame, text="Prédire avec Random Forest", command=predict_forest).grid(row=0, column=1, padx=10)
ttk.Button(btn_frame, text="Diagramme Régression", command=afficher_diagramme_regression).grid(row=0, column=2, padx=10)
ttk.Button(btn_frame, text="Diagramme Forest", command=afficher_diagramme_forest).grid(row=0, column=3, padx=10)
ttk.Button(btn_frame, text="Réinitialiser", command=reset_fields).grid(row=0, column=4, padx=10)

# Résultats
result_linear = tk.Label(root, text="", bg="#1e1e1e", fg="white", font=("Segoe UI", 14))
result_linear.pack(pady=(10, 5))

result_forest = tk.Label(root, text="", bg="#1e1e1e", fg="white", font=("Segoe UI", 14))
result_forest.pack(pady=(5, 10))

# Deux zones pour les graphiques
chart_frame_linear = tk.Frame(root, bg="#1e1e1e")
chart_frame_linear.pack(pady=5)

chart_frame_forest = tk.Frame(root, bg="#1e1e1e")
chart_frame_forest.pack(pady=5)

# Lancement
root.mainloop()
