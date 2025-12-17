# created on Jul 14, 2025

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt  # 导入绘图库

from utils import (
    create_ry_matrix,
    _create_full_system_gate,
    create_custom_block_diagonal,
    build_layered_product_matrix,
    build_global_qnn
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
    def __init__(self, num_qnn_layers: int, num_qnn_blocks: int, m: int = 4, num_qnn_qubits: int = 2):
        """
        初始化QNN分类器模型。
        """
        super().__init__()

        # --- 模型超参数 ---
        self.m = m
        self.num_qnn_qubits = num_qnn_qubits
        self.total_qubits = m + num_qnn_qubits  # 4 + 2 = 6
        self.num_qnn_layers = num_qnn_layers
        self.num_qnn_blocks = num_qnn_blocks

        # --- 定义可训练参数 ---
        # 1. 用于 build_layered_product_matrix 的参数 (alpha)
        num_alpha_params = 2 ** m - 1  # 2**4 - 1 = 15
        self.alpha = nn.Parameter(torch.rand(num_alpha_params, dtype=torch.float32) * np.pi * 2)

        # 2. 用于QNN块的参数 (thetas)
        num_params_per_qnn = num_qnn_layers * num_qnn_qubits
        self.qnn_thetas = nn.Parameter(torch.rand(self.num_qnn_blocks, num_params_per_qnn,
                                                  dtype=torch.float32) * np.pi * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播过程。
        x: 输入数据，形状为 (batch_size, 4)。
        """
        device = self.alpha.device
        dtype = torch.cfloat

        batch_outputs = []

        for i in range(x.shape[0]):
            x_sample = x[i].to(device=device, dtype=dtype)

            # 1. 构建输入量子态
            # initial_state_m = torch.zeros(2 ** self.m, device=device, dtype=dtype)
            # initial_state_m[0] = 1.0
            # initial_state = torch.kron(initial_state_m, x_sample)  # (16,) ⊗ (4,) -> (64,)

            initial_state = torch.zeros(2 ** self.total_qubits, device=device, dtype=dtype)
            initial_state[0] = 1.0

            # 2. 构建第一层酉矩阵 U_layer1
            entangling_block = build_layered_product_matrix(self.m, self.alpha.to(dtype))
            identity_qnn = torch.eye(2 ** self.num_qnn_qubits, device=device, dtype=dtype)
            u_layer1 = torch.kron(entangling_block, identity_qnn)  # (16x16) ⊗ (4x4) -> (64x64)

            # 3. 计算中间量子态
            intermediate_state = u_layer1 @ initial_state

            # 4. 构建块对角QNN层 U_qnn_layer
            qnn_matrices = []
            for j in range(self.num_qnn_blocks):
                thetas_for_qnn = self.qnn_thetas[j].view(self.num_qnn_layers, self.num_qnn_qubits)
                qnn_unit = build_global_qnn(self.num_qnn_qubits, self.num_qnn_layers, thetas_for_qnn.to(dtype))
                qnn_matrices.append(qnn_unit)

            # create_custom_block_diagonal 期望有 2**m = 16 个矩阵
            u_qnn_layer = create_custom_block_diagonal(self.m, qnn_matrices)  # -> (64x64)

            # 5. 计算最终量子态
            final_state = u_qnn_layer @ intermediate_state

            # 6. 测量并计算输出
            measurement_index = self.m  # 索引为4 (第5个量子比特)
            z_op = get_pauli_z_operator(self.total_qubits, measurement_index, device)
            exp_val = torch.real(torch.vdot(final_state, z_op.to(dtype) @ final_state))
            # output_prob = (exp_val + 1) / 2
            # batch_outputs.append(output_prob)
            batch_outputs.append(exp_val)

        return torch.stack(batch_outputs)


# ==============================================================================
# ==============================================================================

def run_gradient_experiment(num_repetitions=50, learning_rate=0.001):
    """
    主执行函数，用于进行梯度收集实验。
    """
    print("1. 准备MNIST数据集 (类别 9 和 6)...")
    transform = transforms.Compose([
        transforms.Resize((2, 2)),
        transforms.ToTensor()
    ])
    full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    def get_class_indices(dataset, classes):
        indices = []
        for i in range(len(dataset)):
            if dataset[i][1] in classes:
                indices.append(i)
        return indices

    train_indices = get_class_indices(full_train_dataset, [9, 6])
    train_dataset = Subset(full_train_dataset, train_indices)

    def preprocess_data(subset):
        images, labels = [], []
        for img, label in subset:
            img_flat = img.view(-1)
            norm = torch.linalg.vector_norm(img_flat)
            img_normalized = img_flat / norm if norm > 0 else torch.zeros_like(img_flat)
            images.append(img_normalized)
            labels.append(0.0 if label == 9 else 1.0)
        return torch.stack(images), torch.tensor(labels, dtype=torch.float32)

    X_train, y_train = preprocess_data(train_dataset)
    print(f"数据集准备完毕。使用 {len(y_train)} 个样本。输入向量维度: {X_train.shape[1]}")

    DEVICE = torch.device("cpu")
    print(f"将使用设备: {DEVICE}")

    # 定义实验参数
    D_indices = sorted({1, 2, 4})
    L_indices = range(1, 17, 1)  # 从1到16

    # 初始化用于存储结果的数组
    # 维度: 4 (D_index) x 16 (L_index) x 50 (repetitions)
    collected_gradients = np.zeros((len(D_indices), len(L_indices), num_repetitions))

    print("\n2. 开始梯度收集实验...")

    # 只取一个数据样本用于所有实验
    input_sample, label_sample = X_train[0:1].to(DEVICE), y_train[0:1].to(DEVICE)
    criterion = nn.MSELoss()

    for i, D_index in enumerate(D_indices):
        for j, L_index in enumerate(L_indices):
            m_value = max(1, (L_index - 1).bit_length())
            print(f"--- Running for D_index={D_index}, L_index={L_index} ---")
            for k in tqdm(range(num_repetitions), desc=f"D={D_index}, L={L_index} (m={m_value})", unit="rep"):
                # --- 每次循环都重新初始化模型和优化器 ---
                model = MNIST_QNN_Classifier(
                    num_qnn_layers=D_index,
                    num_qnn_blocks=L_index,
                    m=m_value  # 显式设置 m=4
                ).to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # --- 执行单次前向和后向传播 ---
                optimizer.zero_grad()
                outputs = model(input_sample).squeeze()
                # loss = criterion(outputs, label_sample)
                outputs.backward()

                # --- 提取、打印并保存第一个参数的梯度 ---
                # 确保梯度存在 (如果L_index=0，则qnn_thetas可能未使用)
                if model.qnn_thetas.grad is not None and model.qnn_thetas.grad.numel() > 0:
                    first_grad_value = model.qnn_thetas.grad[0, 0].item()
                    collected_gradients[i, j, k] = first_grad_value

                    # 按要求打印梯度值
                    # 为了避免刷屏，只在第一次重复时打印
                    if k == 0:
                        print(f"  Rep {k + 1}: Extracted grad for D={D_index}, L={L_index} -> {first_grad_value:.8f}")

                else:
                    # 如果没有梯度，则记录为NaN
                    collected_gradients[i, j, k] = np.nan
                    if k == 0:
                        print(f"  Rep {k + 1}: Grad for D={D_index}, L={L_index} is None.")

    print("\n实验完成！")
    print(f"已收集所有梯度。最终数组维度: {collected_gradients.shape}")

    # 你可以在这里添加代码来保存或进一步分析 `collected_gradients` 数组
    file_name = "collected_gradients.npy"
    np.save(file_name, collected_gradients)
    print(f"\n梯度数据已成功保存到: {file_name}")
    return collected_gradients, D_indices, L_indices


# ==============================================================================
# ==============================================================================
if __name__ == '__main__':
    # 设置为 True: 运行耗时的梯度收集实验并保存结果。
    # 设置为 False: 直接从 "collected_gradients.npy" 文件加载数据并绘图。
    RUN_NEW_EXPERIMENT = True


    if RUN_NEW_EXPERIMENT:
        print("模式: 运行新实验并保存数据。")
        # 1. 运行实验并获取/保存梯度数据
        repetitions = 1000  # 注意：1000次重复会花费很长时间！
        collected_gradients, D_indices, L_indices = run_gradient_experiment(
            num_repetitions=repetitions
        )
    else:
        print("模式: 从文件加载数据并绘图。")
        # 1. 从文件加载预先计算的梯度数据
        data_file = "collected_gradients.npy"
        try:
            collected_gradients = np.load(data_file)
            print(f"成功从 '{data_file}' 加载数据。")
            # 根据数组形状和实验设置，重新生成索引
            D_indices = sorted({1, 2, 4})
            L_indices = range(1, collected_gradients.shape[1] + 1)  # 从形状推断L范围
        except FileNotFoundError:
            print(f"错误: 数据文件 '{data_file}' 未找到。")
            print("请先将 RUN_NEW_EXPERIMENT 设置为 True 运行一次以生成数据文件。")
            exit()  # 如果文件不存在，则退出程序

    # 2. 计算方差
    # 沿着最后一个轴 (axis=2)，即50次重复的轴，计算方差
    gradient_variances = np.var(collected_gradients, axis=2)
    gradient_mean = np.mean(collected_gradients, axis=2)
    print(f"\n计算得到的均值: {gradient_mean}")  # 应为 (4, 16)
    print(f"\n计算得到的方差数组维度: {gradient_variances.shape}")  # 应为 (4, 16)

    # 3. 绘图
    print("正在生成梯度方差图...")
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用美观的绘图风格
    fig, ax = plt.subplots(figsize=(12, 8))

    # 为每个深度(D)画一条线
    for i, D_val in enumerate(D_indices):
        # 提取当前深度的方差数据 (一行)
        variances_for_D = gradient_variances[i, :]
        ax.plot(
            list(L_indices),  # X 轴: L from 1 to 16
            variances_for_D,  # Y 轴: 计算出的方差
            marker='o',  # 在每个数据点上加一个圆圈标记
            linestyle='-',  # 用实线连接数据点
            label=f'Depth={D_val}'  # 图例标签
        )

    # 4. 美化图表
    ax.set_title('Gradient Variance vs. Number of QNN Blocks (L)', fontsize=16)
    ax.set_xlabel('Number of QNN Blocks (L)', fontsize=12)
    ax.set_ylabel("Variance of θ_0,0 Gradient", fontsize=12)

    # 考虑使用对数坐标轴，因为方差可能会跨越多个数量级
    ax.set_yscale('log')
    ax.set_ylabel("Variance of θ_0,0 Gradient", fontsize=12)

    ax.set_xticks(list(L_indices))  # 确保X轴上每个整数点都有刻度
    plt.xticks(rotation=45)  # 如果刻度太多，可以旋转一下
    ax.legend(title='QNN Depth (D)', fontsize=10)  # 显示图例
    fig.tight_layout()  # 调整布局以防标签重叠

    file_name = "gradient_variance_vs_L_plot.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"\n图表已成功保存为: {file_name}")

    # 5. 显示图表
    # plt.show()
