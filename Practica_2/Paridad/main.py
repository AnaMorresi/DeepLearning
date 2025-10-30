import numpy as np
from activations import Tanh, ReLU, Sigmoid
from layers import Dense, Skip
from models import Network, SkipNetwork
from losses import MSE
from optimizers import GD
import matplotlib
matplotlib.use("TkAgg")  # Fuerza el backend Tkinter
import matplotlib.pyplot as plt
import os

os.makedirs("Paridad/figuras", exist_ok=True)

# # Datos Paridad
# N=3
# X = np.array(np.meshgrid(*([[-1,1]]*N))).T.reshape(-1,N)
# y = np.prod(X,axis=1).reshape(-1, 1)
# print(y.shape)

# ### Arquitectura: N -> N' -> 1
# N_prima=10
# model = Network()
# model.add(Dense(N, N_prima, Tanh))
# model.add(Dense(N_prima, 1, Tanh))
# model.set_loss(MSE())
# model.set_optimizer(GD(lr=0.01))
# loss_hist, acc_hist = model.train(X,y,epochs=1000)

# y_pred = model.predict(X)
# print("\nPredicciones finales:")
# print(np.round(y_pred, 3))
# print("Salida esperada:")
# print(y)

# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# # ---- Gráfico 1: Loss ----
# axs[0].plot(loss_hist, color='blue')
# axs[0].set_title("Función de costo (Loss)")
# axs[0].set_xlabel("Épocas")
# axs[0].set_ylabel("Loss (MSE)")
# axs[0].grid(True)
# # ---- Gráfico 2: Accuracy ----
# axs[1].plot(acc_hist, color='green')
# axs[1].set_title("Precisión (Accuracy)")
# axs[1].set_xlabel("Épocas")
# axs[1].set_ylabel("Accuracy")
# axs[1].set_ylim(-0.05, 1.05)
# axs[1].grid(True)
# # Ajustes de espacio y título general
# plt.suptitle("Evolución del entrenamiento de la red Paridad", fontsize=14)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig("Paridad/figuras/entrenamiento_Paridad.png", dpi=300)
# plt.show()



# ======== PARÁMETROS ========
N = 6                     # número de entradas fijo
N_primas = [2, 4, 8, 16]  # número de neuronas ocultas
epochs = 1000
lr = 0.01
runs = 100                # número de corridas por modelo

# ======== DATOS DE PARIDAD ========
X = np.array(np.meshgrid(*([[-1, 1]] * N))).T.reshape(-1, N)
y = np.prod(X, axis=1).reshape(-1, 1)

# ======== ENTRENAMIENTOS ========
results = {}
for Np in N_primas:
    print(f"\nEntrenando redes con N={N}, N'={Np} ...")
    all_loss = []
    all_acc = []

    for run in range(runs):
        model = Network()
        model.add(Dense(N, Np, Tanh))
        model.add(Dense(Np, 1, Tanh))
        model.set_loss(MSE())
        model.set_optimizer(GD(lr=lr))
        loss_hist, acc_hist = model.train(X, y, epochs=epochs)
        all_loss.append(loss_hist)
        all_acc.append(acc_hist)

    # Calcular promedio y desvío estándar
    all_loss = np.array(all_loss)
    all_acc = np.array(all_acc)
    mean_loss = np.mean(all_loss, axis=0)
    std_loss = np.std(all_loss, axis=0)
    mean_acc = np.mean(all_acc, axis=0)
    std_acc = np.std(all_acc, axis=0)

    results[Np] = {
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "mean_acc": mean_acc,
        "std_acc": std_acc
    }

# ======== GRÁFICOS ========
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# --- Gráfico 1: Loss ---
for Np in N_primas:
    mean_loss = results[Np]["mean_loss"]
    std_loss = results[Np]["std_loss"]
    axs[0].plot(mean_loss, label=f"N'={Np}")
    axs[0].fill_between(range(epochs),
                        mean_loss - std_loss,
                        mean_loss + std_loss,
                        alpha=0.2)
axs[0].set_title(f"Loss promedio para N={N} (100 corridas)")
axs[0].set_xlabel("Épocas")
axs[0].set_ylabel("MSE")
axs[0].legend()
axs[0].grid(True)

# --- Gráfico 2: Accuracy ---
for Np in N_primas:
    mean_acc = results[Np]["mean_acc"]
    std_acc = results[Np]["std_acc"]
    axs[1].plot(mean_acc, label=f"N'={Np}")
    axs[1].fill_between(range(epochs),
                        mean_acc - std_acc,
                        mean_acc + std_acc,
                        alpha=0.2)
axs[1].set_title(f"Accuracy promedio para N={N} (100 corridas)")
axs[1].set_xlabel("Épocas")
axs[1].set_ylabel("Precisión")
axs[1].set_ylim(-0.05, 1.05)
axs[1].legend()
axs[1].grid(True)

plt.suptitle(f"Efecto del tamaño de la capa oculta (N={N})", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"Paridad/figuras/comparativa_N{N}_promedios.png", dpi=300)
plt.show()

