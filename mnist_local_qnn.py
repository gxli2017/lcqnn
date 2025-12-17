import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm  # 导入tqdm用于显示进度条
import warnings

# 导入之前定义的所有工具函数
from utils import (
    create_ry_matrix,
    _create_full_system_gate,
    create_custom_block_diagonal,
    build_layered_product_matrix,
    build_global_qnn,
    build_local_qnn_0,
    build_local_qnn_1
)

warnings.filterwarnings("ignore", category=UserWarning)


# --- 辅助函数：创建测量算子 ---
def get_pauli_z_operator(num_qubits, qubit_index, device):
    """为n比特系统创建在指定比特上的Pauli-Z测量算子 Z_i。"""
    I = torch.eye(2, device=device, dtype=torch.float32)
    Z = torch.tensor([[1, 0], [0, -1]], device=device, dtype=torch.float32)

    op_list = [I] * num_qubits
    op_list[qubit_index] = Z

    return _create_full_system_gate(op_list, num_qubits)


# ==============================================================================
# QNN 模型定义
# ==============================================================================

class MNIST_QNN_Classifier(nn.Module):
    def __init__(self, num_qnn_layers: int, num_qnn_blocks: int, m: int = 2, num_qnn_qubits: int = 10):
        """
        初始化QNN分类器模型。
        """
        super().__init__()

        # --- 模型超参数 ---
        self.m = m
        self.num_qnn_qubits = num_qnn_qubits
        self.total_qubits = m + num_qnn_qubits  # 2 + 10 = 12
        self.num_qnn_layers = num_qnn_layers
        self.num_qnn_blocks = num_qnn_blocks

        # --- 定义可训练参数 ---
        # 1. 用于 build_layered_product_matrix 的参数 (alpha)
        num_alpha_params = 2 ** m - 1  # 2**3 - 1 = 7
        self.alpha = nn.Parameter(torch.rand(num_alpha_params, dtype=torch.float32) * np.pi * 2)

        # 2. 用于QNN块的参数 (thetas)
        num_params_per_qnn = num_qnn_layers * num_qnn_qubits
        self.qnn_thetas = nn.Parameter(torch.rand(self.num_qnn_blocks, num_params_per_qnn,
                                                  dtype=torch.float32) * np.pi * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播过程。
        x: 输入数据，形状为 (batch_size, 64)。
        """
        # 获取设备和数据类型
        device = self.alpha.device
        # 量子计算通常使用复数
        dtype = torch.cfloat

        batch_outputs = []

        # 为了简洁，我们使用batch_size=1的循环处理
        # 复杂化：可以对整个批次进行矢量化操作，但会增加代码复杂度
        for i in range(x.shape[0]):
            x_sample = x[i].to(device=device, dtype=dtype)  # 64维复数向量

            # --- 1. 构建输入量子态 ---
            # a. 创建 m=3 比特的 |0...0> 态
            initial_state_m = torch.zeros(2 ** self.m, device=device, dtype=dtype)
            initial_state_m[0] = 1.0

            # b. 张量积组合成9比特输入态: |000> ⊗ |image_vector>
            # x_sample 已经是单位向量，可以直接用作量子态振幅
            initial_state = torch.kron(initial_state_m, x_sample)

            # --- 2. 构建第一层酉矩阵 U_layer1 ---
            # a. 构建 m=3 的分层矩阵
            entangling_block = build_layered_product_matrix(self.m, self.alpha.to(dtype))

            # b. 创建一个6比特的单位矩阵
            identity_qnn = torch.eye(2 ** self.num_qnn_qubits, device=device, dtype=dtype)

            # c. U_layer1 = U_entangle(3-qubit) ⊗ I(6-qubit)
            u_layer1 = torch.kron(entangling_block, identity_qnn)

            # --- 3. 计算中间量子态 ---
            intermediate_state = u_layer1 @ initial_state

            # --- 4. 构建块对角QNN层 U_qnn_layer ---
            qnn_matrices = []
            for j in range(self.num_qnn_blocks):
                thetas_for_qnn = self.qnn_thetas[j].view(self.num_qnn_layers, self.num_qnn_qubits)
                # 检查 j 的奇偶性
                if j % 2 == 0:
                    # j 是偶数 (0, 2, 4, ...)，使用 build_local_qnn_0
                    qnn_unit = build_local_qnn_0(self.num_qnn_qubits, self.num_qnn_layers, thetas_for_qnn.to(dtype))
                else:
                    # j 是奇数 (1, 3, 5, ...)，使用 build_local_qnn_1
                    qnn_unit = build_local_qnn_1(self.num_qnn_qubits, self.num_qnn_layers, thetas_for_qnn.to(dtype))
                qnn_matrices.append(qnn_unit)

            u_qnn_layer = create_custom_block_diagonal(self.m, qnn_matrices)

            # --- 5. 计算最终量子态 ---
            final_state = u_qnn_layer @ intermediate_state

            # --- 6. 测量并计算10分类的输出 ---
            # 在10个量子比特上分别测量Pauli-Z算符的期望值。
            # 每一个期望值 <Z_i> (范围在[-1, 1]) 将作为对应类别的logit。
            # 总共有 self.total_qubits = 10 个量子比特。
            #
            output_logits = []
            for qubit_idx in list(range(self.m, self.total_qubits)):
                # 1. 为当前比特(qubit_idx)创建Pauli-Z测量算子
                z_op = get_pauli_z_operator(self.total_qubits, qubit_idx, device)

                # 2. 计算期望值 <ψ_final| Z_i |ψ_final>
                exp_val = torch.real(torch.vdot(final_state, z_op.to(dtype) @ final_state))

                # 3. 将这个期望值作为一个类的原始输出(logit)
                output_logits.append(exp_val)

            # 将10个logits堆叠成一个 [10,] 的张量
            output_10_classes = torch.stack(output_logits)
            # ====================================================================

            batch_outputs.append(output_10_classes)
            # print(batch_outputs)

        # 最终返回一个 [batch_size, 10] 的张量
        return torch.stack(batch_outputs)


# ==============================================================================
# 主函数：训练和评估
# ==============================================================================

def main(num_qnn_blocks: int = 2, num_qnn_layers: int = 2, epochs: int = 20, learning_rate: float = 0.01):
    """主执行函数"""
    num_qnn_qubits = 4
    # --- 1. 数据准备 (MNIST 0 vs 1) ---
    print(f"1. 准备MNIST数据集 (类别 0 到 {num_qnn_qubits - 1})...")
    transform = transforms.Compose([
        transforms.Resize((4, 4)),
        transforms.ToTensor()
    ])
    full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def get_class_indices(dataset, classes):
        indices = []
        for i in range(len(dataset)):
            if dataset[i][1] in classes:
                indices.append(i)
        return indices

    classes_to_select = list(range(num_qnn_qubits))
    train_indices = get_class_indices(full_train_dataset, classes_to_select)
    test_indices = get_class_indices(full_test_dataset, classes_to_select)
    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)

    def preprocess_data(subset):
        images, labels = [], []
        for img, label in subset:
            img_flat = img.view(-1)
            norm = torch.linalg.vector_norm(img_flat)
            img_normalized = img_flat / norm if norm > 0 else torch.zeros_like(img_flat)
            images.append(img_normalized)

            # 将标签 3 映射到 0.0，标签 6 映射到 1.0
            # labels.append(0.0 if label == 9 else 1.0)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)

    X_train, y_train = preprocess_data(train_dataset)
    X_test, y_test = preprocess_data(test_dataset)

    # X_train = X_train[:1000]
    # y_train = y_train[:1000]
    # X_test = X_test[:400]
    # y_test = y_test[:400]

    print("\n--- 数据集统计 ---")

    # 使用 torch.bincount 高效地统计每个类别的数量
    train_counts = torch.bincount(y_train)
    test_counts = torch.bincount(y_test)

    print("训练集 (y_train) 中各个类别的数目:")
    for i, count in enumerate(train_counts):
        if i in classes_to_select:
            print(f"  类别 {i}: {count} 个")

    print("\n测试集 (y_test) 中各个类别的数目:")
    for i, count in enumerate(test_counts):
        if i in classes_to_select:
            print(f"  类别 {i}: {count} 个")

    print("-------------------\n")
    print("数据准备和统计完成。")

    print(f"训练集大小: {len(y_train)}, 测试集大小: {len(y_test)}")
    print(f"输入向量维度: {X_train.shape[1]}")

    # --- 2. 模型、损失函数和优化器 ---
    print(f"\n2. 初始化模型 (num_qnn_blocks={num_qnn_blocks}) 和优化器...")
    DEVICE = torch.device("cpu")
    print(f"将使用设备: {DEVICE}")

    model = MNIST_QNN_Classifier(
        num_qnn_layers=num_qnn_layers,
        num_qnn_blocks=num_qnn_blocks,
        m=2,
        num_qnn_qubits=num_qnn_qubits  # 32x32图片 -> 1024维向量 -> 10个量子比特
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 3. 训练循环 ---
    print("\n3. 开始训练...")
    batch_size = 1
    train_tensor_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)

    # 定义评估间隔
    evaluation_interval = 5000
    # 数据集总数
    num_train_samples = len(train_loader)

    # 初始化列表以存储最后一个epoch的评估准确率
    last_epoch_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # 使用tqdm创建带描述的进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="sample")

        # 评估计数器
        eval_count_in_epoch = 1

        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 更新进度条的后缀信息，显示当前平均loss
            progress_bar.set_postfix(loss=f'{running_loss / (i + 1):.4f}')

            # 在每个评估间隔点，进行测试集评估
            # 检查是否到达评估点 (i+1 是当前处理的样本数)
            # 最后一个batch也进行评估
            # 检查是否到达评估点
            if (i + 1) % evaluation_interval == 0 or (i + 1) == num_train_samples:
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test.to(DEVICE)).squeeze()
                    predicted = torch.argmax(test_outputs, dim=1)
                    correct = (predicted == y_test.to(DEVICE)).sum().item()
                    test_accuracy = 100 * correct / len(y_test)

                # 【新增修改 2/3】如果这是最后一个epoch，则记录准确率
                if epoch == epochs - 1:
                    last_epoch_accuracies.append(test_accuracy)

                print(f"\nEpoch [{epoch + 1}], Step [{i + 1}/{num_train_samples}] (评估点 {eval_count_in_epoch}) - "
                      f"Test Acc: {test_accuracy:6.2f}%")

                eval_count_in_epoch += 1
                model.train()

        # 每个epoch结束后，打印最终的平均loss
        avg_loss = running_loss / num_train_samples
        print(f"Epoch [{epoch + 1}/{epochs}] 训练完成。 平均Loss: {avg_loss:.4f}")

    print("\n训练完成！")

    # 计算并打印最后一个epoch评估点的均值和标准差
    if last_epoch_accuracies:
        acc_tensor = torch.tensor(last_epoch_accuracies)
        mean_acc = acc_tensor.mean().item()
        std_acc = acc_tensor.std().item()
        print(f"\n最后一个Epoch的所有评估点 Test Accuracy 统计:")
        print(f"  - 均值: {mean_acc:.2f}%")
        print(f"  - 标准差: {std_acc:.2f}%")

    # --- 最终评估模型 ---
    print(f"\n4. 在测试集上评估最终模型 (num_qnn_blocks={num_qnn_blocks})...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test.to(DEVICE)).squeeze()
        predicted_classes = torch.argmax(test_outputs, dim=1)
        correct = (predicted_classes == y_test.to(DEVICE)).sum().item()
        total = y_test.size(0)
        accuracy = 100 * correct / total
        print(f"最终测试集准确率: {accuracy:.2f}%")


if __name__ == '__main__':
    # 您可以在这里修改参数来调用和测试
    for D_index in sorted({1, 2, 4, 8}):
        for L_index in sorted({1, 2, 4}):
            print(f"num_qnn_blocks: {L_index}, num_qnn_layers: {D_index}")
            main(
                num_qnn_blocks=L_index,
                num_qnn_layers=D_index,
                epochs=2,  # 总epoch数
                learning_rate=0.008
            )
