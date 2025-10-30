import numpy as np
from activations import Tanh, ReLU, Sigmoid, Lineal
from layers import Dense, Skip
from models import Network, SkipNetwork
import metric
from losses import MSE
from optimizers import SGD
import matplotlib
matplotlib.use("TkAgg")  # Fuerza el backend Tkinter
import matplotlib.pyplot as plt
import os

os.makedirs("XOR/figuras", exist_ok=True)

# Datos XOR (-1 y 1)
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y = np.array([[1], [-1], [-1], [1]])

### Arquitectura 1: 2 -> 2 -> 1
model = Network()
model.add(Dense(2, 2, Tanh))
model.add(Dense(2, 1, Tanh))

# Configurar pérdida y optimizador
model.set_loss(MSE())
model.set_optimizer(SGD(lr=0.1))

# Entrenar
loss_hist, acc_hist,_,_ = model.train(X, y, epochs=1000)

# Predicción final
y_pred = model.predict(X)
print("\nPredicciones finales:")
print(np.round(y_pred, 3))
print("Salida esperada:")
print(y)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# ---- Gráfico 1: Loss ----
axs[0].plot(loss_hist, color='blue')
axs[0].set_title("Función de costo (Loss)")
axs[0].set_xlabel("Épocas")
axs[0].set_ylabel("Loss (MSE)")
axs[0].grid(True)
# ---- Gráfico 2: Accuracy ----
axs[1].plot(acc_hist, color='green')
axs[1].set_title("Precisión (Accuracy)")
axs[1].set_xlabel("Épocas")
axs[1].set_ylabel("Accuracy")
axs[1].set_ylim(-0.05, 1.05)
axs[1].grid(True)
# Ajustes de espacio y título general
plt.suptitle("Evolución del entrenamiento de la red XOR", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("XOR/figuras/entrenamiento_XOR_Arquitectura1.png", dpi=300)
plt.show()

### Arquitectura 2: 2 -> 2 -> 1 con skip desde la entrada a la salida
# Capas
hidden = Dense(input_size=2, output_size=1, activation=Tanh)
output = Skip(input_size=2, hidden_size=1, output_size=1, activation=Tanh)
# Red con skip
net = SkipNetwork(hidden, output)
net.set_loss(MSE())
net.set_optimizer(SGD(lr=0.01))
# Entrenar
loss_hist, acc_hist = net.train(X, y, epochs=1000)

# Predicción final
y_pred = net.forward(X)
print("\nPredicciones finales:")
print(np.round(y_pred, 3))
print("Salida esperada:")
print(y)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# ---- Gráfico 1: Loss ----
axs[0].plot(loss_hist, color='blue')
axs[0].set_title("Función de costo (Loss)")
axs[0].set_xlabel("Épocas")
axs[0].set_ylabel("Loss (MSE)")
axs[0].grid(True)
# ---- Gráfico 2: Accuracy ----
axs[1].plot(acc_hist, color='green')
axs[1].set_title("Precisión (Accuracy)")
axs[1].set_xlabel("Épocas")
axs[1].set_ylabel("Accuracy")
axs[1].set_ylim(-0.05, 1.05)
axs[1].grid(True)
# Ajustes de espacio y título general
plt.suptitle("Evolución del entrenamiento de la red XOR", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("XOR/figuras/entrenamiento_XOR_Arquitectura2.png", dpi=300)
plt.show()

# MEDIA DE CORRER VARIAS VECES EL MODELO 1
### Arquitectura 1: 2 -> 2 -> 1
N_run=500
predictions = np.zeros((N_run, len(X)))
for i in range(N_run):
    model = Network()
    model.add(Dense(2, 2, Tanh))
    model.add(Dense(2, 1, Tanh))
    model.set_loss(MSE())
    model.set_optimizer(SGD(lr=0.1))
    loss_hist, acc_hist,_,_ = model.train(X, y, epochs=1500)

    # Predicción final
    y_pred = model.forward(X)
    predictions[i] = y_pred.flatten()

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()
labels = ["(1, 1)", "(1, -1)", "(-1, 1)", "(-1, -1)"]
true_vals = [1, -1, -1, 1]
for i in range(4):
    axs[i].hist(predictions[:, i], bins=15, color="royalblue", edgecolor="white", alpha=0.8)
    axs[i].axvline(true_vals[i], color="red", linestyle="--", label="Valor esperado")
    axs[i].set_title(f"Entrada {labels[i]}")
    axs[i].set_xlabel("Predicción")
    axs[i].set_ylabel("Frecuencia")
    axs[i].legend()
plt.suptitle(f"Distribución de salidas del XOR ({N_run} corridas)", fontsize=14)

y_pred_sign = np.sign(predictions)
acc_per_run = np.mean(y_pred_sign == y.T, axis=1)  # precisión por red
print("Accuracy promedio:", np.mean(acc_per_run))
print("Accuracy mínima:", np.min(acc_per_run))
plt.gcf().text(
    0.75, 0.95,  # x, y en coordenadas de la figura (0 a 1)
    f"Acc promedio: {np.mean(acc_per_run):.2f}\nAcc mínima: {np.min(acc_per_run):.2f}",
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("XOR/figuras/histogramas_XOR_Arquitectura1.png", dpi=300)
plt.show()

# MEDIA DE CORRER VARIAS VECES EL MODELO 2
### Arquitectura 2: 2 -> 2 -> 1 con skip desde la entrada a la salida
N_run=500
predictions = np.zeros((N_run, len(X)))
for i in range(N_run):
    hidden = Dense(input_size=2, output_size=1, activation=Tanh)
    output = Skip(input_size=2, hidden_size=1, output_size=1, activation=Tanh)
    net = SkipNetwork(hidden, output)
    net.set_loss(MSE())
    net.set_optimizer(SGD(lr=0.01))
    loss_hist, acc_hist = net.train(X, y, epochs=1500)

    # Predicción final
    y_pred = net.forward(X)
    predictions[i] = y_pred.flatten()

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()
labels = ["(1, 1)", "(1, -1)", "(-1, 1)", "(-1, -1)"]
true_vals = [1, -1, -1, 1]
for i in range(4):
    axs[i].hist(predictions[:, i], bins=15, color="royalblue", edgecolor="white", alpha=0.8)
    axs[i].axvline(true_vals[i], color="red", linestyle="--", label="Valor esperado")
    axs[i].set_title(f"Entrada {labels[i]}")
    axs[i].set_xlabel("Predicción")
    axs[i].set_ylabel("Frecuencia")
    axs[i].legend()
plt.suptitle(f"Distribución de salidas del XOR ({N_run} corridas)", fontsize=14)

y_pred_sign = np.sign(predictions)
acc_per_run = np.mean(y_pred_sign == y.T, axis=1)  # precisión por red
print("Accuracy promedio:", np.mean(acc_per_run))
print("Accuracy mínima:", np.min(acc_per_run))
plt.gcf().text(
    0.75, 0.95,  # x, y en coordenadas de la figura (0 a 1)
    f"Acc promedio: {np.mean(acc_per_run):.2f}\nAcc mínima: {np.min(acc_per_run):.2f}",
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("XOR/figuras/histogramas_XOR_Arquitectura2.png", dpi=300)
plt.show()
