# created on Jul 11, 2025

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 导入之前定义的所有工具函数
from utils import (
    create_ry_matrix,
    _create_full_system_gate,
    create_custom_block_diagonal,
    build_layered_product_matrix,
    build_global_qnn
)


# --- 辅助函数：创建测量算子 ---
def get_pauli_z_operator(num_qubits, qubit_index, device):
    """为n比特系统创建在指定比特上的Pauli-Z测量算子 Z_i。"""
    I = torch.eye(2, device=device, dtype=torch.float32)
    Z = torch.tensor([[1, 0], [0, -1]], device=device, dtype=torch.float32)

    op_list = [I] * num_qubits
    op_list[qubit_index] = Z

    return _create_full_system_gate(op_list, num_qubits)


# --- 定义完整的QNN分类器模型 ---
class IrisQNNClassifier(nn.Module):
    def __init__(self, num_qnn_layers: int, m: int = 3, num_qnn_qubits: int = 4):
        """
        初始化QNN分类器模型。

        参数:
        num_qnn_layers (int): 每个QNN块的深度。
        m (int): 控制分层矩阵和块对角矩阵的参数。
        num_qnn_qubits (int): 每个QNN块的量子比特数。
        """
        super().__init__()

        # --- 模型超参数 ---
        self.m = m
        self.num_qnn_qubits = num_qnn_qubits
        self.total_qubits = m + num_qnn_qubits  # 3 + 4 = 7

        # --- 定义可训练参数 ---

        # 1. 用于 build_layered_product_matrix 的参数 (alpha)
        num_alpha_params = 2 ** m - 1  # 2**3 - 1 = 7
        self.alpha = nn.Parameter(torch.rand(num_alpha_params) * 6.28)

        # 2. 用于3个4比特QNN的参数 (thetas)
        # 每个QNN有 num_qnn_layers * num_qnn_qubits 个参数
        num_qnn_outputs = 4  # 对应Iris的3个分类
        num_params_per_qnn = num_qnn_layers * num_qnn_qubits
        self.qnn_thetas = nn.Parameter(torch.rand(num_qnn_outputs, num_params_per_qnn) * 6.28)

        self.num_qnn_layers = num_qnn_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播过程。

        参数:
        x (torch.Tensor): 输入数据，形状为 (batch_size, 4)。

        返回:
        torch.Tensor: 模型的输出，形状为 (batch_size, 3)。
        """
        # 获取设备和数据类型信息
        device = self.alpha.device
        dtype = self.alpha.dtype

        # 初始化一个列表来收集批处理中每个样本的输出
        batch_outputs = []

        # 对批处理中的每个数据点进行处理
        for i in range(x.shape[0]):
            x_sample = x[i]

            # --- 1. 数据预处理和编码 ---
            # a. 归一化到 [0, pi/2]
            # 注意：在训练流程中，我们已经对整个数据集做了归一化
            # 这里直接将[0,1]的输入映射到[0, pi/2]
            angles = x_sample * (np.pi / 0.5)

            # b. 将输入作用于Ry门，并用_create_full_system_gate组合
            ry_gates = [create_ry_matrix(angle) for angle in angles]
            encoding_matrix = _create_full_system_gate(ry_gates, self.num_qnn_qubits)

            # --- 2. 构建第一层酉矩阵 U_layer1 ---
            # a. 构建 m=3 的分层矩阵
            entangling_block = build_layered_product_matrix(self.m, self.alpha)

            # b. 通过张量积组合成一个 2**7 x 2**7 的大矩阵
            # U_layer1 = U_entangle(3-qubit) ⊗ U_encode(4-qubit)
            u_layer1 = torch.kron(entangling_block, encoding_matrix)

            # --- 3. 计算中间量子态 ---
            # a. 初始化7比特的全0量子态
            initial_state = torch.zeros(2 ** self.total_qubits, device=device, dtype=dtype)
            initial_state[0] = 1.0

            # b. 作用第一层矩阵
            intermediate_state = u_layer1 @ initial_state

            # --- 4. 构建块对角QNN层 U_qnn_layer ---
            # a. 为3个输出类别分别构建一个4比特的QNN矩阵
            qnn_matrices = []
            for j in range(self.qnn_thetas.shape[0]):  # 循环3次
                # 重塑参数以匹配 build_global_qnn 的输入要求
                thetas_for_qnn = self.qnn_thetas[j].view(self.num_qnn_layers, self.num_qnn_qubits)
                qnn_unit = build_global_qnn(self.num_qnn_qubits, self.num_qnn_layers, thetas_for_qnn)
                qnn_matrices.append(qnn_unit)

            # b. 使用 create_custom_block_diagonal 创建块对角矩阵
            # m=3 -> 8个槽位, L=3 -> 3个QNN + 5个单位矩阵填充
            u_qnn_layer = create_custom_block_diagonal(self.m, qnn_matrices)

            # --- 5. 计算最终量子态 ---
            final_state = u_qnn_layer @ intermediate_state

            # --- 6. 测量并计算输出 ---
            # a. 在第4,5,6比特（索引为3,4,5）上计算Pauli-Z期望
            # 注意: 第一个比特是索引0，所以第4比特是索引3
            measurement_indices = [3, 4, 5]
            expectations = []
            for qubit_idx in measurement_indices:
                z_op = get_pauli_z_operator(self.total_qubits, qubit_idx, device)

                # <ψ|Z|ψ> = ψ† Z ψ
                exp_val = torch.real(torch.vdot(final_state, z_op @ final_state))
                expectations.append(exp_val)

            # b. 将期望值[-1, 1]映射到[0, 1]并组合成最终输出
            output_probs = (torch.stack(expectations) + 1) / 2
            batch_outputs.append(output_probs)

        return torch.stack(batch_outputs)


# --- 训练和评估的主流程 ---
if __name__ == '__main__':
    # --- 1. 数据准备 ---
    print("1. 准备Iris数据集...")
    iris = load_iris()
    X = iris.data
    y = iris.target

    # a. 将数据归一化到 [0, 1] 范围
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # b. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.5, random_state=7564, stratify=y
    )

    # c. 转换为PyTorch张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    print("number of train:", len(y_train_t), "number of test:", len(y_test_t))

    # d. 对标签进行One-Hot编码
    y_train_onehot = nn.functional.one_hot(y_train_t, num_classes=3).float()

    # --- 2. 模型、损失函数和优化器 ---
    print("2. 初始化模型和优化器...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {DEVICE}")

    # 模型超参数
    QNN_DEPTH = 8  # 每个QNN块的深度

    model = IrisQNNClassifier(num_qnn_layers=QNN_DEPTH).to(DEVICE)
    criterion = nn.MSELoss()  # 使用均方误差损失
    # criterion = nn.CrossEntropyLoss()  # 使用均方误差损失

    optimizer = optim.Adam(model.parameters(), lr=0.008)

    # 将数据移动到指定设备
    X_train_t = X_train_t.to(DEVICE)
    # y_train_t = y_train_t.to(DEVICE)
    y_train_onehot = y_train_onehot.to(DEVICE)
    X_test_t = X_test_t.to(DEVICE)
    y_test_t = y_test_t.to(DEVICE)

    # --- 3. 训练循环 ---
    print("3. 开始训练...")
    epochs = 30
    batch_size = 1

    for epoch in range(epochs):
        model.train()
        # 创建数据加载器以进行批处理训练
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_onehot)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        if (epoch + 1) % 1 == 0:
            # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

            model.eval()  # 切换到评估模式
            with torch.no_grad():
                test_outputs = model(X_test_t)
                predicted = torch.argmax(test_outputs, dim=1)
                correct = (predicted == y_test_t).sum().item()
                total = y_test_t.size(0)
                test_accuracy = 100 * correct / total

            # 获取参数值
            alpha_vals = model.alpha.detach().cpu().numpy()
            qnn_thetas_norm = torch.norm(model.qnn_thetas).item()

            # 格式化alpha以便于阅读
            alpha_str = np.array2string(alpha_vals, precision=3, floatmode='fixed', sign=' ')

            # 打印所有信息
            print(f"Epoch [{epoch + 1:2d}/{epochs}], Loss: {avg_loss:.4f}, Test Acc: {test_accuracy:6.2f}%")
            # print(f"    Params -> Alpha: {alpha_str} | QNN Thetas (Norm): {qnn_thetas_norm:.4f}")

    print("训练完成！")

    # --- 4. 评估模型 ---
    print("\n4. 在测试集上评估模型...")
    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 关闭梯度计算
        test_outputs = model(X_test_t)
        # 获取概率最高的类别作为预测结果
        predicted_classes = torch.argmax(test_outputs, dim=1)

        correct = (predicted_classes == y_test_t).sum().item()
        total = y_test_t.size(0)
        accuracy = 100 * correct / total

        print(f"测试集上的准确率: {accuracy:.2f}%")
