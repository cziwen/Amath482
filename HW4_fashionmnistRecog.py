import numpy as np
import torch
from torch import nn
import tqdm

import time

import torchvision
import matplotlib.pyplot as plt
from torch.optim import RMSprop
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# 定义数据预处理
transform = transforms.Compose ([
    transforms.ToTensor (),
    transforms.Normalize ((0.1307,), (0.3081,))
])

# 加载 FashionMNIST 训练 & 测试数据
fashion_train_dataset = torchvision.datasets.FashionMNIST (root='hw4data/', train=True, download=False,
                                                           transform=transform)
fashion_test_dataset = torchvision.datasets.FashionMNIST (root='hw4data/', train=False, download=False,
                                                          transform=transform)

# 加载 MNIST 训练 & 测试数据
mnist_train_dataset = torchvision.datasets.MNIST (root='hw4data/', train=True, download=False, transform=transform)
mnist_test_dataset = torchvision.datasets.MNIST (root='hw4data/', train=False, download=False, transform=transform)

# 划分 FashionMNIST 训练集的 10% 作为验证集
train_indices, val_indices, _, _ = train_test_split (
    range (len (fashion_train_dataset)),
    fashion_train_dataset.targets,
    stratify=fashion_train_dataset.targets,
    test_size=0.1
)

# 划分 FashionMNIST 训练集的 10% 作为验证集
mnist_train_indices, mnist_val_indices, _, _ = train_test_split (
    range (len (mnist_train_dataset)),
    mnist_train_dataset.targets,
    stratify=mnist_train_dataset.targets,
    test_size=0.1
)

# 生成训练 & 验证集的子集
fashion_train_split = Subset (fashion_train_dataset, train_indices)
fashion_val_split = Subset (fashion_train_dataset, val_indices)

mnist_train_split = Subset (mnist_train_dataset, mnist_train_indices)
mnist_val_split = Subset (mnist_train_dataset, mnist_val_indices)

# 设置批量大小
train_batch_size = 512
test_batch_size = 256

# 创建 DataLoader
fashion_train_batches = DataLoader (fashion_train_split, batch_size=train_batch_size, shuffle=True)
fashion_val_batches = DataLoader (fashion_val_split, batch_size=train_batch_size, shuffle=True)
fashion_test_batches = DataLoader (fashion_test_dataset, batch_size=test_batch_size, shuffle=True)

mnist_train_batches = DataLoader (mnist_train_split, batch_size=train_batch_size, shuffle=True)
mnist_val_batches = DataLoader (mnist_val_split, batch_size=train_batch_size, shuffle=True)
mnist_test_batches = DataLoader (mnist_test_dataset, batch_size=test_batch_size, shuffle=True)


# 打印 batch 数量
# num_fashion_train_batches = len (fashion_train_batches)
# num_fashion_val_batches = len (fashion_val_batches)
# num_fashion_test_batches = len (fashion_test_batches)
#
# num_mnist_train_batches = len (mnist_train_batches)
# num_mnist_test_batches = len (mnist_test_batches)
#
# print (f"FashionMNIST 训练批次数: {num_fashion_train_batches}")
# print (f"FashionMNIST 验证批次数: {num_fashion_val_batches}")
# print (f"FashionMNIST 测试批次数: {num_fashion_test_batches}")
#
# print (f"MNIST 训练批次数: {num_mnist_train_batches}")
# print (f"MNIST 测试批次数: {num_mnist_test_batches}")


# =============== 定义 FCN ====================


class ACAIGFCN (nn.Module):
    def __init__ (self,
                  input_dim=784,
                  hidden_dim=[256, 128],  # 多层隐藏单元
                  output_dim=10,
                  learning_rate=0.001,
                  dropout_rate=0.0,
                  optimizer='SGD',
                  init_method='Xavier',
                  normalization='none'):
        super (ACAIGFCN, self).__init__ ()

        # ---- STEP 1: Build layers with optional normalization ----
        layers = []

        # First Layer
        prev_dim = input_dim
        for h_dim in hidden_dim:
            layers.append (nn.Linear (prev_dim, h_dim))
            if normalization == 'batchnorm':
                layers.append (nn.BatchNorm1d (h_dim))
            elif normalization == 'layernorm':
                layers.append (nn.LayerNorm (h_dim))
            layers.append (nn.ReLU ())
            layers.append (nn.Dropout (p=dropout_rate))
            prev_dim = h_dim  # 更新上一层的输出维度

        # Output Layer
        layers.append (nn.Linear (prev_dim, output_dim))

        self.model = nn.Sequential (*layers)

        # ---- STEP 2: Initialize weights ----
        for layer in self.model:
            if isinstance (layer, nn.Linear):
                if init_method == 'Random':
                    nn.init.normal_ (layer.weight, mean=0.0, std=0.01)
                elif init_method == 'Xavier':
                    nn.init.xavier_normal_ (layer.weight)
                elif init_method == 'Kaiming':
                    nn.init.kaiming_uniform_ (layer.weight, nonlinearity='relu')

        self.learning_rate = learning_rate
        self.loss_func = nn.CrossEntropyLoss ()

        # ---- STEP 3: Choose optimizer ----
        if optimizer == 'RMSProp':
            self.optimizer = torch.optim.RMSprop (self.parameters (), lr=self.learning_rate)
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam (self.parameters (), lr=self.learning_rate)
        else:  # 默认使用 SGD
            self.optimizer = torch.optim.SGD (self.parameters (), lr=self.learning_rate)

    def forward (self, x):
        # Flatten input from (batch, 1, 28, 28) -> (batch, 784)
        x = x.view (x.size (0), -1)
        return self.model (x)

    def train_model (self, train_batches, num_epochs=10, validate_batches=None):
        train_loss_list = []
        accuracy_list = []

        for epoch in tqdm.trange (num_epochs):
            total_loss = 0
            self.train ()
            for train_features, train_labels in train_batches:
                # Forward pass
                outputs = self (train_features)
                loss = self.loss_func (outputs, train_labels)

                # Backprop
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
                print (
                    f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.6f}")

        # # Plot the training loss curve
        # plt.figure (figsize=(8, 5))
        # plt.plot (range (1, num_epochs + 1), train_loss_list, marker='o', linestyle='-', label='Training Loss')
        # plt.xlabel ("Epochs")
        # plt.ylabel ("Loss")
        # plt.title (f"Training Loss Curve {label}")
        # plt.legend ()
        # plt.grid (True)
        # plt.show ()
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

        val_acc = correct / total
        return val_acc


def plot_loss_curves (loss_curve_dict):
    """
    绘制多个训练损失曲线，并叠加在同一个图上。

    参数:
        loss_curve_dict (dict):
            - key: 代表不同实验或模型的标签 (str)
            - value: 对应的 loss 曲线数据 (list 或 np.array)，长度等于训练 epoch 数
    """
    plt.figure (figsize=(8, 5))

    # 生成颜色序列
    colors = plt.cm.tab10 (np.linspace (0, 1, len (loss_curve_dict)))

    for i, (label, loss_curve) in enumerate (loss_curve_dict.items ()):
        plt.plot (range (1, len (loss_curve) + 1), loss_curve, marker='o', linestyle='-', label=label,
                  color=colors[i % 10])

    plt.xlabel ("Epochs")
    plt.ylabel ("Loss")
    plt.title ("Training Loss Curves")
    plt.legend ()
    plt.grid (True)
    plt.show ()


def plot_accuracy_curves (accuracy_curve_dict):
    """
    绘制多个准确率曲线，并叠加在同一个图上。

    参数:
        accuracy_curve_dict (dict):
            - key: 代表不同实验或模型的标签 (str)
            - value: 对应的 accuracy 曲线数据 (list 或 np.array)，长度等于训练 epoch 数
    """
    plt.figure (figsize=(8, 5))

    # 生成颜色序列
    colors = plt.cm.tab10 (np.linspace (0, 1, len (accuracy_curve_dict)))

    for i, (label, acc_curve) in enumerate (accuracy_curve_dict.items ()):
        plt.plot (range (1, len (acc_curve) + 1), acc_curve, marker='o', linestyle='-', label=label,
                  color=colors[i % 10])

    plt.xlabel ("Epochs")
    plt.ylabel ("Accuracy")
    plt.title ("Accuracy Curves")
    plt.legend ()
    plt.grid (True)
    plt.show ()


# ============ Base Line Config
def train_validate (num_epochs=20, hidden_dim=[112, 96], lr=0.1, optimizer=None, init_method=None, norm_method=None,
                    dropout=0.0, train_batch=fashion_train_batches, validate_batch=fashion_val_batches,
                    test_batch=fashion_test_batches):
    model = ACAIGFCN (
        input_dim=784,
        hidden_dim=hidden_dim,
        output_dim=10,
        learning_rate=lr,
        dropout_rate=dropout,
        optimizer=optimizer,
        init_method=init_method,
        normalization=norm_method
    )

    start_time = time.time()
    loss_curve, acc_curve = model.train_model (train_batch, num_epochs=num_epochs,
                                               validate_batches=validate_batch)

    end_time = time.time()
    time_usage = end_time - start_time

    val_acc = model.validate_model (validate_batch)
    test_acc = model.validate_model (test_batch)

    return val_acc, test_acc, loss_curve, acc_curve, time_usage


# ============================= baseline =======================
# lr = 0.1
# num_epochs = 20
# results_dict = {}
# loss_curve_dict = {}
# acc_curve_dict = {}
# val_acc, test_acc, loss_curve, acc_curve = train_validate (num_epochs=num_epochs, lr=lr)
# results_dict["Baseline Config"] = f"{val_acc * 100:.2f}, {test_acc * 100:.2f}"
# loss_curve_dict["Baseline Config"] = loss_curve
# acc_curve_dict["Baseline Config"] = acc_curve
#
# plot_loss_curves(loss_curve_dict)
# plot_accuracy_curves(acc_curve_dict)
#
# print(results_dict)


# ============================= 找最好的 optimizer =======================
# optimizers = ['SGD', 'RMSProp', 'Adam']
# lr = 0.01
# num_epochs = 20
# results_dict = {}
# loss_curve_dict = {}
# acc_curve_dict = {}
#
# for optimizer in optimizers:
#     val_acc, test_acc, loss_curve, acc_curve = train_validate (num_epochs=num_epochs, lr=lr, optimizer=optimizer)
#     results_dict[optimizer] = f"{val_acc * 100:.2f}, {test_acc * 100:.2f}"
#     loss_curve_dict[optimizer] = loss_curve
#     acc_curve_dict[optimizer] = acc_curve
#
# plot_loss_curves(loss_curve_dict)
# plot_accuracy_curves(acc_curve_dict)
#
# print(results_dict)


# ============================= 找最好的 drop out =======================
# dropout = [0.0, 0.2, 0.4, 0.6, 0.8]
# lr = 0.01
# num_epochs = 20
# results_dict = {}
# loss_curve_dict = {}
# acc_curve_dict = {}
#
# for dropout in dropout:
#     val_acc, test_acc, loss_curve, acc_curve = train_validate (num_epochs=num_epochs, lr=lr, optimizer="Adam", dropout=dropout)
#     results_dict[dropout] = f"{val_acc * 100:.2f}, {test_acc * 100:.2f}"
#     loss_curve_dict[dropout] = loss_curve
#     acc_curve_dict[dropout] = acc_curve
#
# plot_loss_curves(loss_curve_dict)
# plot_accuracy_curves(acc_curve_dict)
#
# print(results_dict)


# ============================= 找最好的 init =======================
# init_methods = ['Random', 'Xavier', 'Kaiming']
# lr = 0.01
# num_epochs = 20
# results_dict = {}
# loss_curve_dict = {}
# acc_curve_dict = {}
#
# for init in init_methods:
#     val_acc, test_acc, loss_curve, acc_curve = train_validate (num_epochs=num_epochs, lr=lr, optimizer="Adam", dropout=0.0, init_method=init)
#     results_dict[init] = f"{val_acc * 100:.2f}, {test_acc * 100:.2f}"
#     loss_curve_dict[init] = loss_curve
#     acc_curve_dict[init] = acc_curve
#
# plot_loss_curves(loss_curve_dict)
# plot_accuracy_curves(acc_curve_dict)
#
# print(results_dict)


# ============================= 找最好的 norm =======================
# normalizations = ['none', 'batchnorm', 'layernorm']
# lr = 0.01
# num_epochs = 20
# results_dict = {}
# loss_curve_dict = {}
# acc_curve_dict = {}
#
# for norm in normalizations:
#     val_acc, test_acc, loss_curve, acc_curve = (
#         train_validate (num_epochs=num_epochs, lr=lr, optimizer="Adam", dropout=0.0, init_method='Random', norm_method=norm))
#     results_dict[norm] = f"{val_acc * 100:.2f}, {test_acc * 100:.2f}"
#     loss_curve_dict[norm] = loss_curve
#     acc_curve_dict[norm] = acc_curve
#
# plot_loss_curves (loss_curve_dict)
# plot_accuracy_curves (acc_curve_dict)
#
# print (results_dict)

# ================== Extra credit, hyper tuning for 90% and 98% ====================
# opt = "Adam"
# init = "Random"
# norm = "layernorm"
#
# lr = 0.0004
# num_epochs = 100
# results_dict = {}
# loss_curve_dict = {}
# acc_curve_dict = {}
#
# # 训练 fashion
# val_acc, test_acc, loss_curve, acc_curve = (
#         train_validate (num_epochs=num_epochs, lr=lr, optimizer="Adam", dropout=0.2, init_method='Random'
#                         , norm_method=norm, train_batch=fashion_train_batches, validate_batch=fashion_val_batches))
#
# results_dict["FashionMNIST"] = f"{val_acc * 100:.2f}, {test_acc * 100:.2f}"
# loss_curve_dict["FashionMNIST"] = loss_curve
# acc_curve_dict["FashionMNIST"] = acc_curve
#
#
#
# # 训练 MNIST
# val_acc, test_acc, loss_curve, acc_curve = (
#         train_validate (num_epochs=num_epochs, lr=lr, optimizer="Adam", dropout=0.2, init_method='Random'
#                         , norm_method=norm, train_batch=mnist_train_batches, validate_batch=mnist_val_batches
#                         , test_batch=mnist_test_batches))
#
# results_dict["MNIST"] = f"{val_acc * 100:.2f}, {test_acc * 100:.2f}"
# loss_curve_dict["MNIST"] = loss_curve
# acc_curve_dict["MNIST"] = acc_curve
#
#
#
# print(results_dict)
# plot_loss_curves (loss_curve_dict)
# plot_accuracy_curves (acc_curve_dict)

# 定义我的最强 SVM 分类器
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def svm_Classify (batch1, batch2, k=59):
    """
    训练 SVM 分类器，支持 PCA 降维，接受两个 batch 作为输入
    """
    # 处理 DataLoader 的情况
    if isinstance (batch1, torch.utils.data.DataLoader):
        batch1 = next (iter (batch1))
    if isinstance (batch2, torch.utils.data.DataLoader):
        batch2 = next (iter (batch2))

    X_train, y_train = batch1
    X_test, y_test = batch2

    # 转换为 numpy 数组（如果数据是 PyTorch 张量）
    if isinstance (X_train, torch.Tensor):
        X_train = X_train.numpy ()
        X_test = X_test.numpy ()
        y_train = y_train.numpy ()
        y_test = y_test.numpy ()

    # 调整形状：展平成 (num_samples, num_features)
    X_train = X_train.reshape (X_train.shape[0], -1)  # (batch, 784)
    X_test = X_test.reshape (X_test.shape[0], -1)  # (batch, 784)

    # PCA
    pca = PCA (n_components=k)
    pca.fit (X_train)
    X_train_pca = pca.transform (X_train)
    X_test_pca = pca.transform (X_test)

    # 训练 SVM
    svm_clf = SVC (kernel='rbf')
    svm_clf.fit (X_train_pca, y_train)

    # 预测
    y_pred_train = svm_clf.predict (X_train_pca)
    y_pred_test = svm_clf.predict (X_test_pca)

    train_acc = accuracy_score (y_train, y_pred_train)
    test_acc = accuracy_score (y_test, y_pred_test)

    print (f"\n======== SVM Classification on Fashion Mnist ========")
    print (f"Train Accuracy: {train_acc:.4f}")
    print (f"Test Accuracy: {test_acc:.4f}")

# svm_Classify(fashion_train_batches, fashion_test_batches)
