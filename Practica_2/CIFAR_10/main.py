import numpy as np
from activations import Tanh, ReLU, Sigmoid, Lineal
from layers import Dense
from models import Network
import metric
from losses import MSE, MSE_L2
from optimizers import Mini_Batch
import matplotlib
matplotlib.use("TkAgg")  # Fuerza el backend Tkinter
import matplotlib.pyplot as plt
import os
import keras

os.makedirs("CIFAR_10/figuras", exist_ok=True)

def y_ones(labels):
    y=np.zeros((labels.shape[0], len(np.unique(labels))), dtype=np.float32)
    y[np.arange(labels.shape[0]) ,labels.flatten()] = 1.0
    return y

# Datos CIFAR_10
### Importo Datos
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
#Aplanar datos para usar modelo
x_train = x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
x_test = x_test.reshape((x_test.shape[0], np.prod(x_test.shape[1:])))
y_train = y_ones(y_train)
y_test = y_ones(y_test)
# Creo datos validacion de train
permut = np.random.permutation(y_train.shape[0])
y_train_permut=y_train[permut]
x_train_permut=x_train[permut]
N=int(y_train.shape[0]*0.2)
y_val = y_train_permut[:N]
x_val = x_train_permut[:N]
y_train = y_train_permut[N:]
x_train = x_train_permut[N:]
print(y_test.shape[0],y_val.shape[0],y_train.shape[0])

### Arquitectura: 3072 -> 100 -> 100 -> 10
model = Network()
model.add(Dense(x_train.shape[1], 100, Sigmoid))
model.add(Dense(100, 100, Sigmoid))
model.add(Dense(100, 10, Lineal))
model.set_loss(MSE_L2())
model.set_optimizer(Mini_Batch(lr=0.01,batch_size=500,lreg=0.001))
loss_hist, acc_hist, loss_hist_val, acc_hist_val = model.train(x_train,y_train,epochs=4,x_val=x_val,y_val=y_val)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# ---- Gráfico 1: Loss ----
axs[0].plot(loss_hist, color='blue', label='train')
axs[0].plot(loss_hist_val, color='red', label='validation')
axs[0].set_title("Función de costo (Loss)")
axs[0].set_xlabel("Épocas")
axs[0].set_ylabel("Loss (MSE)")
axs[0].legend()
axs[0].grid(True)
# ---- Gráfico 2: Accuracy ----
axs[1].plot(acc_hist, color='green', label='train')
axs[1].plot(acc_hist_val, color='orange', label='validation')
axs[1].set_title("Precisión (Accuracy)")
axs[1].set_xlabel("Épocas")
axs[1].set_ylabel("Accuracy")
axs[1].set_ylim(-0.05, 1.05)
axs[1].legend()
axs[1].grid(True)
# Ajustes de espacio y título general
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle("Evolución del entrenamiento de la red CIFAR_10", fontsize=14)
plt.savefig("CIFAR_10/figuras/entrenamiento_CIFAR_10.png", dpi=300)
plt.show()

# Provamos en el test
y_predict_test=model.predict(x_test)
acc_test=metric.accuracy(y_predict_test,y_test)
print('Accuracy en test:',acc_test)
y_predict_val=model.predict(x_val)
acc_val=metric.accuracy(y_predict_val,y_val)
print('Accuracy en val:',acc_val)
y_predict_train=model.predict(x_train)
acc_train=metric.accuracy(y_predict_train,y_train)
print('Accuracy en train:',acc_train)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cifar10_labels = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}
labels = [cifar10_labels[i] for i in range(10)]
# Calcular etiquetas predichas y verdaderas
y_true = np.argmax(y_test, axis=1)
y_predict_train = np.argmax(y_predict_test, axis=1)
# Matriz de confusión
cm = confusion_matrix(y_true, y_predict_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
plt.title("Matriz de confusión - CIFAR-10 (Test)")
plt.savefig("CIFAR_10/figuras/matconf_CIFAR_10.png", dpi=300)
plt.show()