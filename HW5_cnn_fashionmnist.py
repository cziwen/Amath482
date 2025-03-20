import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import torchvision
import torchvision.transforms as transforms
from numpy.f2py.crackfortran import endifs
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from HW4_fashionmnistRecog import (ACAIGFCN, fashion_train_batches
, fashion_val_batches, train_validate, plot_loss_curves, plot_accuracy_curves, fashion_test_batches)




opt = "Adam"
init = "Random"
norm = "layernorm"

lr = 0.001
dr = 0.01
num_epochs = 20
results_dict = {}
loss_curve_dict = {}
acc_curve_dict = {}
time_dict = {}

# ==================== task 1 100k less weight, 88% test Accuracy =======================
hidden_dim = [112, 96]
val_acc, test_acc, loss_curve, acc_curve, time_usage = (
    train_validate (num_epochs=num_epochs, hidden_dim=hidden_dim, lr=lr, optimizer="Adam", dropout=dr,
                    init_method='Random'
                    , norm_method=norm, train_batch=fashion_train_batches, validate_batch=fashion_val_batches))

results_dict["100k"] = f"val acc: {val_acc * 100:.2f}, test acc: {test_acc * 100:.2f}"
loss_curve_dict["100k"] = loss_curve
acc_curve_dict["100k"] = acc_curve
time_dict["100k"] = time_usage

# ===================== task 2 比较不同的 weight 的 FCN =========================
# 50k
hidden_dim = [60, 32]
val_acc, test_acc, loss_curve, acc_curve, time_usage = (
    train_validate (num_epochs=num_epochs, hidden_dim=hidden_dim, lr=lr, optimizer="Adam", dropout=dr,
                    init_method='Random'
                    , norm_method=norm, train_batch=fashion_train_batches, validate_batch=fashion_val_batches))

results_dict["50k"] = f"val acc: {val_acc * 100:.2f}, test acc: {test_acc * 100:.2f}"
loss_curve_dict["50k"] = loss_curve
acc_curve_dict["50k"] = acc_curve
time_dict["50k"] = time_usage

# 200k
hidden_dim = [224, 32]
val_acc, test_acc, loss_curve, acc_curve, time_usage = (
    train_validate (num_epochs=num_epochs, hidden_dim=hidden_dim, lr=lr, optimizer="Adam", dropout=dr,
                    init_method='Random'
                    , norm_method=norm, train_batch=fashion_train_batches, validate_batch=fashion_val_batches))

results_dict["200k"] = f"val acc: {val_acc * 100:.2f}, test acc: {test_acc * 100:.2f}"
loss_curve_dict["200k"] = loss_curve
acc_curve_dict["200k"] = acc_curve
time_dict["200k"] = time_usage

print (results_dict)
print (time_dict)
plot_loss_curves (loss_curve_dict)
plot_accuracy_curves (acc_curve_dict)


# ================================= 定义 CNN ============================

class ACAIConvNet (nn.Module):
    def __init__ (self,
                  input_channels=1,
                  input_size=(28, 28),
                  conv_layers=[(32, 3, 1), (64, 3, 1)],
                  fc_layers=[128, 64],
                  output_dim=10,
                  learning_rate=0.001,
                  dropout_rate=0.0,
                  optimizer='SGD',
                  init_method='Xavier',
                  normalization='none'):
        super (ACAIConvNet, self).__init__ ()

        self.learning_rate = learning_rate
        self.loss_func = nn.CrossEntropyLoss ()

        # ---- STEP 1: Build Convolutional Layers ----
        layers = []
        prev_channels = input_channels

        for out_channels, kernel_size, stride in conv_layers:
            layers.append (nn.Conv2d (prev_channels, out_channels, kernel_size, stride, padding=1))

            # 可选：根据 `normalization` 来添加对应层
            if normalization == 'batchnorm':
                layers.append (nn.BatchNorm2d (out_channels))
            elif normalization == 'layernorm':
                layers.append (nn.GroupNorm (1, out_channels))

            layers.append (nn.ReLU ())
            layers.append (nn.MaxPool2d (2, 2))  # 减半高宽
            layers.append (nn.Dropout (p=dropout_rate))

            prev_channels = out_channels

        self.conv_layers = nn.Sequential (*layers)


        with torch.no_grad ():
            test_input = torch.randn (1, input_channels, *input_size)
            test_output = self.conv_layers (test_input)
            flattened_size = test_output.view (1, -1).size (1)

        # ---- STEP 2: Fully Connected Layers ----
        fc_layers_list = []
        prev_dim = flattened_size
        for h_dim in fc_layers:
            fc_layers_list.append (nn.Linear (prev_dim, h_dim))
            fc_layers_list.append (nn.ReLU ())
            fc_layers_list.append (nn.Dropout (p=dropout_rate))
            prev_dim = h_dim

        fc_layers_list.append (nn.Linear (prev_dim, output_dim))
        self.fc_layers = nn.Sequential (*fc_layers_list)

        # ---- STEP 3: Initialize Weights ----
        for layer in self.modules ():
            if isinstance (layer, nn.Conv2d) or isinstance (layer, nn.Linear):
                if init_method == 'Random':
                    nn.init.normal_ (layer.weight, mean=0.0, std=0.01)
                elif init_method == 'Xavier':
                    nn.init.xavier_normal_ (layer.weight)
                elif init_method == 'Kaiming':
                    nn.init.kaiming_uniform_ (layer.weight, nonlinearity='relu')

        # ---- STEP 4: Choose Optimizer ----
        if optimizer == 'RMSProp':
            self.optimizer = optim.RMSprop (self.parameters (), lr=self.learning_rate)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam (self.parameters (), lr=self.learning_rate)
        else:
            self.optimizer = optim.SGD (self.parameters (), lr=self.learning_rate)

    def forward (self, x):
        x = self.conv_layers (x)
        x = x.view (x.size (0), -1)  # Flatten
        x = self.fc_layers (x)
        return x

    def train_model (self, train_batches, num_epochs=10, validate_batches=None):
        train_loss_list = []
        accuracy_list = []

        for epoch in tqdm.trange (num_epochs):
            total_loss = 0
            self.train ()
            for train_features, train_labels in train_batches:
                outputs = self (train_features)
                loss = self.loss_func (outputs, train_labels)

                self.optimizer.zero_grad ()
                loss.backward ()
                self.optimizer.step ()

                total_loss += loss.item ()

            avg_loss = total_loss / len (train_batches)
            train_loss_list.append (avg_loss)

            if validate_batches is not None:
                val_acc = self.validate_model (validate_batches)
                accuracy_list.append (val_acc)
                print (
                    f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.6f}, Validation Accuracy: {val_acc:.4f}")
            else:
                print (f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.6f}")

        return train_loss_list, accuracy_list

    def validate_model (self, val_batches):
        correct = 0
        total = 0
        self.eval ()
        with torch.no_grad ():
            for val_features, val_labels in val_batches:
                outputs = self (val_features)
                _, predicted = torch.max (outputs, 1)
                correct += (predicted == val_labels).sum ().item ()
                total += val_labels.size (0)
        return correct / total

    def count_params (self):
        return sum (p.numel () for p in self.parameters () if p.requires_grad)


def train_validate_cnn (num_epochs=20,
                        conv_layers=[(32, 3, 1), (64, 3, 1)],
                        fc_layers=[128],
                        lr=0.1,
                        optimizer=None,
                        init_method=None,
                        norm_method=None,
                        dropout=0.0,
                        train_batch=None,
                        validate_batch=None,
                        test_batch=None,
                        input_channels=1,
                        input_size=(28, 28)):  # <--- 新增这两个参数
    model = ACAIConvNet (
        input_channels=input_channels,
        input_size=input_size,  # <--- 传给 ACAIConvNet
        conv_layers=conv_layers,
        fc_layers=fc_layers,
        output_dim=10,
        learning_rate=lr,
        dropout_rate=dropout,
        optimizer=optimizer,
        init_method=init_method,
        normalization=norm_method
    )

    print (f"weights数量: {model.count_params ()}")

    start_time = time.time ()
    loss_curve, acc_curve = model.train_model (train_batch,
                                               num_epochs=num_epochs,
                                               validate_batches=validate_batch)
    end_time = time.time ()
    time_usage = end_time - start_time

    val_acc = model.validate_model (validate_batch) if validate_batch else None
    test_acc = model.validate_model (test_batch) if test_batch else None

    return val_acc, test_acc, loss_curve, acc_curve, time_usage

# ===================== Task 3 ==========================
opt = "Adam"
init = "Random"
norm = "batchnorm"

lr = 0.001
dr = 0.01
num_epochs = 20
results_dict = {}
loss_curve_dict = {}
acc_curve_dict = {}
time_dict = {}


# ============ 100k ==============
fc_layer = [64, 32]
cov_layer = [(14, 3, 1), (28, 3, 1)]

val_acc, test_acc, loss_curve, acc_curve, time_usage = (
    train_validate_cnn (num_epochs=num_epochs, conv_layers=cov_layer, fc_layers=fc_layer, lr=lr, optimizer=opt,
                        init_method=init, norm_method=norm, dropout=dr, train_batch=fashion_train_batches,
                        validate_batch=fashion_val_batches, test_batch=fashion_test_batches))

results_dict["100k"] = f"val acc: {val_acc * 100:.2f}, test acc: {test_acc * 100:.2f}"
loss_curve_dict["100k"] = loss_curve
acc_curve_dict["100k"] = acc_curve
time_dict["100k"] = time_usage

# ============== 50k =================
fc_layer = [46, 20]
cov_layer = [(6, 3, 1), (21, 3, 1)]

val_acc, test_acc, loss_curve, acc_curve, time_usage = (
    train_validate_cnn (num_epochs=num_epochs, conv_layers=cov_layer, fc_layers=fc_layer, lr=lr, optimizer=opt,
                        init_method=init, norm_method=norm, dropout=dr, train_batch=fashion_train_batches,
                        validate_batch=fashion_val_batches, test_batch=fashion_test_batches))

results_dict["50k"] = f"val acc: {val_acc * 100:.2f}, test acc: {test_acc * 100:.2f}"
loss_curve_dict["50k"] = loss_curve
acc_curve_dict["50k"] = acc_curve
time_dict["50k"] = time_usage

# ================= 20k =================
fc_layer = [26, 12]
cov_layer = [(5, 3, 1), (14, 3, 1)]

val_acc, test_acc, loss_curve, acc_curve, time_usage = (
    train_validate_cnn (num_epochs=num_epochs, conv_layers=cov_layer, fc_layers=fc_layer, lr=lr, optimizer=opt,
                        init_method=init, norm_method=norm, dropout=dr, train_batch=fashion_train_batches,
                        validate_batch=fashion_val_batches, test_batch=fashion_test_batches))

results_dict["20k"] = f"val acc: {val_acc * 100:.2f}, test acc: {test_acc * 100:.2f}"
loss_curve_dict["20k"] = loss_curve
acc_curve_dict["20k"] = acc_curve
time_dict["20k"] = time_usage

# # ========================== 10k =======================
fc_layer = [18, 10]
cov_layer = [(4, 3, 1), (10, 3, 1)]

val_acc, test_acc, loss_curve, acc_curve, time_usage = (
    train_validate_cnn (num_epochs=num_epochs, conv_layers=cov_layer, fc_layers=fc_layer, lr=lr, optimizer=opt,
                        init_method=init, norm_method=norm, dropout=dr, train_batch=fashion_train_batches,
                        validate_batch=fashion_val_batches, test_batch=fashion_test_batches))

results_dict["10k"] = f"val acc: {val_acc * 100:.2f}, test acc: {test_acc * 100:.2f}"
loss_curve_dict["10k"] = loss_curve
acc_curve_dict["10k"] = acc_curve
time_dict["10k"] = time_usage


print (results_dict)
print(time_dict)
plot_loss_curves (loss_curve_dict)
plot_accuracy_curves (acc_curve_dict)
