import numpy as np
import matplotlib.pyplot as plt
import keras

### Funciones varias
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def derivada_sigmoid(x):
    #entrada x=sigmoid(z)
    return x*(1.0-x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True)) # para no tener inf en exponencial
    return exps / np.sum(exps, axis=1, keepdims=True)

def y_ones(labels):
    y=np.zeros((labels.shape[0], len(np.unique(labels))), dtype=np.float32)
    y[np.arange(labels.shape[0]) ,labels.flatten()] = 1.0
    return y

def accuracy(y_pred, y_true):
    pred_labels = np.argmax(y_pred, axis=1) # y_pred: (N,n_class)
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(pred_labels == true_labels)

### Inicializacion pesos
def params_init(input_dim, hidden_dim, output_dim, seed):
    np.random.seed(seed)
    # Inicializacion Glorot
    limit1 = np.sqrt(6 / (input_dim + hidden_dim))
    W1 = np.random.uniform(-limit1,limit1,(input_dim,hidden_dim)).astype(np.float32)
    b1 = np.zeros((1, hidden_dim))
    limit2 = np.sqrt(6 / (hidden_dim + output_dim))
    W2 = np.random.uniform(-limit2,limit2,(hidden_dim,output_dim)).astype(np.float32)
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2

### Forward
def forward(x,W1,b1,W2,b2):
    # x: (N,3072)
    z1 = x.dot(W1) + b1         # (N, hidden)
    a1 = sigmoid(z1)            # (N, hidden)
    z2 = a1.dot(W2) + b2        # (N, output) -> salida lineal
    y_pred = z2
    val_med = (x, z1, a1, z2)
    return y_pred, val_med

### Loss (MSE) + L2 regularization
def loss_MSE(y_pred, y_true):
    return 0.5*np.mean(np.sum((y_pred - y_true)**2, axis=1))

def loss_categorical_cross_entropy(y_pred, y_true):
    y = softmax(y_pred)
    eps=1e-10
    return -np.mean(np.sum(y_true*np.log(y+eps), axis=1))

def loss(y_pred, y_true, W1, W2, lreg, MSE=True, CCE=False):
    # L = (1/(2N)) sum ||y_pred - y||^2 + (lambda/2)*(||W1||^2 + ||W2||^2)
    if MSE:
        data_loss = loss_MSE(y_pred, y_true)
    if CCE:
        data_loss = loss_categorical_cross_entropy(y_pred, y_true)
    reg_loss = 0.5*lreg*(np.sum(W1*W1) + np.sum(W2*W2))    
    return data_loss+reg_loss

### Backpropagation
def backward(y_pred, y_true, val_med, W1, W2, reg_lambda, MSE, CCE):
    x, z1, a1, z2 = val_med
    N = y_true.shape[0]

    if MSE:
        dz2 = (y_pred - y_true) / N                 # (N, output)
    if CCE:
        p = softmax(y_pred)                        # aplicar softmax
        dz2 = (p - y_true) / N                     # derivada de CCE
        
    dW2 = a1.T.dot(dz2) + reg_lambda*W2         # (hidden, output)
    db2 = np.sum(dz2, axis=0, keepdims=True)    # (1, output)

    # da1 = dz2 dot W2^T
    da1 = dz2.dot(W2.T)                         # (N, hidden)
    dz1 = da1*derivada_sigmoid(a1)              # (N, hidden)

    dW1 = x.T.dot(dz1) + reg_lambda*W1          # (input, hidden)
    db1 = np.sum(dz1, axis=0, keepdims=True)    # (1, hidden)

    return dW1, db1, dW2, db2

### Entrenamiento 
def train(x_train,y_train,x_val,y_val,
            epochs=25,batch_size=300,lr=0.01,lreg=1e-4,print_cada=1, MSE=False, CCE=False):
    if not MSE and not CCE:
        MSE=True

    input_dim = x_train.shape[1]
    hidden_dim = 100
    output_dim = y_train.shape[1]

    W1,b1,W2,b2=params_init(input_dim,hidden_dim,output_dim,seed=1234)

    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    num_train = x_train.shape[0]
    num_batches = int(np.ceil(num_train/batch_size))

    for epoch in range(1,epochs+1):
        permut = np.random.permutation(num_train)
        x_train_sh = x_train[permut]
        y_train_sh = y_train[permut]

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_train)
            xb = x_train_sh[start:end]
            yb = y_train_sh[start:end]

            y_pred, val_med = forward(xb, W1, b1, W2, b2)     # FORWARD
            dW1,db1,dW2,db2 = backward(y_pred, yb, val_med, W1, W2, lreg, MSE, CCE)   # BACKWARD

            # Update
            W1 -= lr*dW1
            b1 -= lr*db1
            W2 -= lr*dW2
            b2 -= lr*db2

        # Evaluacion en train completo
        y_train_pred, _ = forward(x_train, W1, b1, W2, b2)
        train_loss = loss(y_train_pred, y_train, W1, W2, lreg, MSE, CCE)
        train_acc = accuracy(y_train_pred, y_train)

        y_val_pred, _ = forward(x_val, W1, b1, W2, b2)
        val_loss = loss(y_val_pred, y_val, W1, W2, lreg, MSE, CCE)
        val_acc = accuracy(y_val_pred, y_val)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % print_cada == 0:
            print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    params = (W1, b1, W2, b2)
    return params, history

### MAIN
if __name__=="__main__":
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

    ### Entrenameinto Red
    params, history = train(x_train,y_train,x_val,y_val,
                            epochs=600,batch_size=500,lr=5e-2,lreg=1e-3,print_cada=5, CCE=True, MSE=False)
    
    ### Graficos
    epochs = np.arange(1, len(history['loss'])+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['loss'], label='train loss')
    plt.plot(epochs, history['val_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss (CCE + L2)')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history['acc'], label='train acc')
    plt.plot(epochs, history['val_acc'], label='validation acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Provamos en el test
    y_predict_test,_=forward(x_test,params[0],params[1],params[2],params[3])
    acc_test=accuracy(y_predict_test,y_test)
    print('Accuracy en test:',acc_test)
    y_predict_val,_=forward(x_val,params[0],params[1],params[2],params[3])
    acc_val=accuracy(y_predict_val,y_val)
    print('Accuracy en val:',acc_val)
    y_predict_train,_=forward(x_train,params[0],params[1],params[2],params[3])
    acc_train=accuracy(y_predict_train,y_train)
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
    y_pred = np.argmax(y_predict_test, axis=1)
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
    plt.title("Matriz de confusión - CIFAR-10 (Test)")
    plt.show()