# created on Jul 10, 2025

import torch
from typing import List

# --- 1. 基础矩阵定义 (常量) ---
# 将它们定义在全局，并设置好默认的 device 和 dtype
# 以便在整个程序中重用
DEVICE = torch.device("cpu")
DTYPE = torch.float32

I = torch.tensor([[1, 0], [0, 1]], dtype=DTYPE, device=DEVICE)
X = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE, device=DEVICE)
# 投影算子 |0><0| 和 |1><1|
P0 = torch.tensor([[1, 0], [0, 0]], dtype=DTYPE, device=DEVICE)
P1 = torch.tensor([[0, 0], [0, 1]], dtype=DTYPE, device=DEVICE)


# 创建 Ry 旋转矩阵
def create_ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    """
    根据给定的角度 theta 张量创建一个 2x2 的、可微分的 Ry 旋转矩阵。
    """
    cos_val = torch.cos(theta / 2)
    sin_val = torch.sin(theta / 2)

    row1 = torch.stack([cos_val, -sin_val])
    row2 = torch.stack([sin_val, cos_val])

    # 将其放在与 theta 相同的设备和类型上
    return torch.stack([row1, row2], dim=0).to(theta)


# 创建自定义的块对角矩阵
def create_custom_block_diagonal(
        m: int,
        matrices: List[torch.Tensor]
) -> torch.Tensor:
    """
    根据输入 m 和一个矩阵列表，创建一个块对角矩阵，并确保梯度流。
    (函数实现同上一个回答，这里为保持完整性而包含)
    """
    num_slots = 2 ** m
    num_provided = len(matrices)

    if num_provided > num_slots:
        raise ValueError(f"提供的矩阵数量 ({num_provided}) 不能超过槽位数量 ({num_slots})。")
    if num_provided == 0 and num_slots > 0:
        raise ValueError("输入矩阵列表为空，无法确定填充矩阵的属性。")

    device = matrices[0].device if num_provided > 0 else torch.device('cpu')
    dtype = matrices[0].dtype if num_provided > 0 else torch.float32
    block_dim = matrices[0].shape[0] if num_provided > 0 else 2

    identity_block = torch.eye(block_dim, device=device, dtype=dtype)
    final_blocks = list(matrices)
    num_to_pad = num_slots - num_provided
    if num_to_pad > 0:
        final_blocks.extend([identity_block] * num_to_pad)

    if not final_blocks:
        return torch.empty(0, 0, device=device, dtype=dtype)

    return torch.block_diag(*final_blocks)


# 构建分层矩阵乘积
def build_layered_product_matrix(m: int, alpha: torch.Tensor) -> torch.Tensor:
    """
    根据输入 m 和参数 alpha 构建一个分层矩阵乘积。

    输出 = M_m @ ... @ M_2 @ M_1, 其中 M_k 是根据 alpha 的一部分
    参数构建并通过张量积扩展到 2**m x 2**m 的矩阵。

    参数:
    m (int): 定义了层数和最终矩阵的维度 (2**m x 2**m)。
    alpha (torch.Tensor): 一个包含 2**m - 1 个参数的一维张量。

    返回:
    torch.Tensor: 最终的 (2**m x 2**m) 矩阵乘积。
    """
    # --- 输入验证 ---
    expected_alpha_size = 2 ** m - 1
    if not isinstance(alpha, torch.Tensor) or alpha.dim() != 1:
        raise TypeError("输入 'alpha' 必须是一个一维的 PyTorch 张量。")
    if alpha.numel() != expected_alpha_size:
        raise ValueError(
            f"对于 m={m}, 'alpha' 张量的长度应为 {expected_alpha_size}, "
            f"但实际为 {alpha.numel()}。"
        )

    # 从输入 alpha 推断设备和数据类型
    device = alpha.device
    dtype = alpha.dtype
    total_dim = 2 ** m

    # 初始化最终结果为单位矩阵，后续将从右侧开始累乘
    # Final = M_m @ ... @ M_2 @ (M_1 @ I)
    result_matrix = torch.eye(total_dim, device=device, dtype=dtype)

    alpha_cursor = 0  # 用于在 alpha 张量上滑动的指针

    # --- 循环构建每一层 M_k 并累乘 ---
    for k in range(1, m + 1):
        # 1. 为当前层 k 准备参数
        num_params_for_layer = 2 ** (k - 1)
        layer_alphas = alpha[alpha_cursor: alpha_cursor + num_params_for_layer]
        alpha_cursor += num_params_for_layer

        # 2. 构建核心矩阵 (Core_k)
        ry_matrices = [create_ry_matrix(angle) for angle in layer_alphas]

        # 对于 k=1, core 是单个Ry矩阵, 其余 k>1 使用 block_diag
        # create_custom_block_diagonal(m=0, matrices=[...]) 等价于直接取那个矩阵
        # 所以我们可以统一这个逻辑
        core_k = create_custom_block_diagonal(m=k - 1, matrices=ry_matrices)

        # 3. 通过张量积 (Kronecker product) 扩展到全维度
        identity_dim = 2 ** (m - k)
        identity_matrix = torch.eye(identity_dim, device=device, dtype=dtype)

        # M_k = Core_k ⊗ I
        m_k = torch.kron(core_k, identity_matrix)

        # 4. 矩阵累乘 (从左侧乘上新的矩阵)
        # 第一次循环: result = M_1 @ I = M_1
        # 第二次循环: result = M_2 @ M_1
        # ...
        # 最后一次循环: result = M_m @ ... @ M_1
        result_matrix = m_k @ result_matrix

    return result_matrix


def _create_full_system_gate(
        gate_list: List[torch.Tensor], num_qubits: int
) -> torch.Tensor:
    """
    通过张量积（Kronecker product）将一个门列表扩展为全系统矩阵。
    """
    if len(gate_list) != num_qubits:
        raise ValueError("门列表的长度必须等于量子比特数。")

    # 从第一个门开始，依次与后续的门进行张量积
    full_matrix = gate_list[0]
    for i in range(1, num_qubits):
        full_matrix = torch.kron(full_matrix, gate_list[i])

    return full_matrix


def create_cnot_matrix(
        num_qubits: int, control: int, target: int, device=DEVICE, dtype=DTYPE
) -> torch.Tensor:
    """
    为 n-量子比特系统创建 CNOT(control, target) 的矩阵表示。
    CNOT = |0><0| ⊗ I + |1><1| ⊗ X
    """
    # 确保基础门与目标设备和类型一致
    P0_d = P0.to(device=device, dtype=dtype)
    P1_d = P1.to(device=device, dtype=dtype)
    I_d = I.to(device=device, dtype=dtype)
    X_d = X.to(device=device, dtype=dtype)

    # 第一部分: 控制位为 |0>
    term1_gates = [I_d] * num_qubits
    term1_gates[control] = P0_d
    term1 = _create_full_system_gate(term1_gates, num_qubits)

    # 第二部分: 控制位为 |1>
    term2_gates = [I_d] * num_qubits
    term2_gates[control] = P1_d
    term2_gates[target] = X_d
    term2 = _create_full_system_gate(term2_gates, num_qubits)

    return term1 + term2


# --- 量子神经网络构建函数 ---

def build_global_qnn(
        num_qubits: int, num_depth: int, thetas: torch.Tensor
) -> torch.Tensor:
    """
    构建一个具有全局纠缠层（循环CNOT）的量子神经网络。

    参数:
    num_qubits (int): 量子比特的数量。
    num_depth (int): 网络的深度（层数）。
    thetas (torch.Tensor): 形状为 (num_depth, num_qubits) 的参数张量。

    返回:
    torch.Tensor: 代表整个量子电路的 (2**num_qubits, 2**num_qubits) 酉矩阵。
    """
    # --- 输入验证 ---
    expected_shape = (num_depth, num_qubits)
    if thetas.shape != expected_shape:
        raise ValueError(f"参数 a 的形状应为 {expected_shape}，但得到 {thetas.shape}")

    device = thetas.device
    dtype = thetas.dtype
    total_dim = 2 ** num_qubits

    # 从单位矩阵开始，它代表一个空的电路
    circuit_matrix = torch.eye(total_dim, device=device, dtype=dtype)

    # 构建全局纠缠层矩阵 (只构建一次，因为在每层都相同)
    entanglement_layer = torch.eye(total_dim, device=device, dtype=dtype)
    for q in range(num_qubits - 1):
        control_qubit = q
        target_qubit = (q + 1) % num_qubits  # 循环连接
        cnot_gate = create_cnot_matrix(num_qubits, control_qubit, target_qubit, device, dtype)
        entanglement_layer = cnot_gate @ entanglement_layer

    # 逐层构建网络
    for d in range(num_depth):
        # a) 构建旋转层
        rotation_gates = [create_ry_matrix(thetas[d, q]) for q in range(num_qubits)]
        rotation_layer = _create_full_system_gate(rotation_gates, num_qubits)

        # b) 将这一整层（旋转+纠缠）应用到总电路中
        # 新的层作用在左边
        layer_matrix = entanglement_layer @ rotation_layer
        circuit_matrix = layer_matrix @ circuit_matrix

    return circuit_matrix


def build_local_qnn_0(
        num_qubits: int, num_depth: int, thetas: torch.Tensor
) -> torch.Tensor:
    """
    构建一个具有 2-local 纠缠层（配对CNOT）的量子神经网络。

    参数:
    num_qubits (int): 量子比特的数量。
    num_depth (int): 网络的深度（层数）。
    thetas (torch.Tensor): 形状为 (num_depth, num_qubits) 的参数张量。

    返回:
    torch.Tensor: 代表整个量子电路的 (2**num_qubits, 2**num_qubits) 酉矩阵。
    """
    # --- 输入验证 ---
    expected_shape = (num_depth, num_qubits)
    if thetas.shape != expected_shape:
        raise ValueError(f"参数 a 的形状应为 {expected_shape}，但得到 {thetas.shape}")

    device = thetas.device
    dtype = thetas.dtype
    total_dim = 2 ** num_qubits

    circuit_matrix = torch.eye(total_dim, device=device, dtype=dtype)

    # 构建 2-local 纠缠层矩阵 (只构建一次)
    entanglement_layer = torch.eye(total_dim, device=device, dtype=dtype)
    # 对相邻的量子比特对进行操作
    for q in range(0, num_qubits - (num_qubits % 2), 2):
        q1, q2 = q, (q + 1) % num_qubits
        # CNOT(q1, q2)
        cnot1 = create_cnot_matrix(num_qubits, q1, q2, device, dtype)
        # print(cnot1)
        entanglement_layer = cnot1 @ entanglement_layer
        # CNOT(q2, q1)
        cnot2 = create_cnot_matrix(num_qubits, q2, q1, device, dtype)
        entanglement_layer = cnot2 @ entanglement_layer

    # 逐层构建网络
    for d in range(num_depth):
        # a) 构建旋转层
        rotation_gates = [create_ry_matrix(thetas[d, q]) for q in range(num_qubits)]
        rotation_layer = _create_full_system_gate(rotation_gates, num_qubits)

        # b) 应用完整的层
        layer_matrix = entanglement_layer @ rotation_layer
        circuit_matrix = layer_matrix @ circuit_matrix

    return circuit_matrix


def build_local_qnn_1(
        num_qubits: int, num_depth: int, thetas: torch.Tensor
) -> torch.Tensor:
    """
    构建一个具有 2-local 纠缠层（配对CNOT）的量子神经网络。

    参数:
    num_qubits (int): 量子比特的数量。
    num_depth (int): 网络的深度（层数）。
    thetas (torch.Tensor): 形状为 (num_depth, num_qubits) 的参数张量。

    返回:
    torch.Tensor: 代表整个量子电路的 (2**num_qubits, 2**num_qubits) 酉矩阵。
    """
    # --- 输入验证 ---
    expected_shape = (num_depth, num_qubits)
    if thetas.shape != expected_shape:
        raise ValueError(f"参数 a 的形状应为 {expected_shape}，但得到 {thetas.shape}")

    device = thetas.device
    dtype = thetas.dtype
    total_dim = 2 ** num_qubits

    circuit_matrix = torch.eye(total_dim, device=device, dtype=dtype)

    # 构建 2-local 纠缠层矩阵 (只构建一次)
    entanglement_layer = torch.eye(total_dim, device=device, dtype=dtype)
    # 对相邻的量子比特对进行操作
    for q in range(1, num_qubits - (num_qubits % 2), 2):
        q1, q2 = q, (q + 1) % num_qubits
        # CNOT(q1, q2)
        # print(q1, q2)
        cnot1 = create_cnot_matrix(num_qubits, q1, q2, device, dtype)
        # print(cnot1)
        entanglement_layer = cnot1 @ entanglement_layer
        # CNOT(q2, q1)
        cnot2 = create_cnot_matrix(num_qubits, q2, q1, device, dtype)
        entanglement_layer = cnot2 @ entanglement_layer

    # 逐层构建网络
    for d in range(num_depth):
        # a) 构建旋转层
        rotation_gates = [create_ry_matrix(thetas[d, q]) for q in range(num_qubits)]
        rotation_layer = _create_full_system_gate(rotation_gates, num_qubits)

        # b) 应用完整的层
        layer_matrix = entanglement_layer @ rotation_layer
        circuit_matrix = layer_matrix @ circuit_matrix

    return circuit_matrix

# --- 使用和梯度验证示例 ---
# if __name__ == '__main__':
#     # 设置 m=3, 因此需要 2**3 - 1 = 7 个 alpha 参数
#     m_val = 3
#
#     # 创建需要梯度的 alpha 参数
#     alpha_params = torch.randn(2 ** m_val - 1, requires_grad=True)
#
#     print(f"--- 示例: m = {m_val} ---")
#     print(f"输入 alpha (需要梯度) 的形状: {alpha_params.shape}\n")
#
#     # 调用主函数
#     final_product = build_layered_product_matrix(m_val, alpha_params)
#
#     # 验证梯度流
#     # 计算一个标量损失并反向传播
#     loss = final_product.sum()
#     loss.backward()
#
#     print(f"最终输出矩阵的形状: {final_product.shape}")
#     print(f"最终输出矩阵 (仅显示前 4x4 部分):\n{final_product.data[:4, :4]}\n")
#
#     print("--- 梯度验证 ---")
#     print("输入 alpha 的梯度是否已计算:", alpha_params.grad is not None)
#
#     if alpha_params.grad is not None:
#         print(f"alpha 梯度的形状: {alpha_params.grad.shape}")
#         # 打印部分梯度以确认它们不是零
#         print(f"alpha 梯度的前4个值: {alpha_params.grad[:4]}")
#         # 断言梯度不为空
#         assert alpha_params.grad is not None
#         # 断言梯度的和不为0，证明有有效的梯度信号
#         assert torch.abs(alpha_params.grad).sum() > 1e-6
#         print("\n结论：梯度已成功从最终矩阵回传至输入的 alpha 张量。")
#
#     # --- Case: num_qubits = 4 ---
#     n_qubits = 2
#     n_depth = 2  # 使用一个较小的深度进行演示
#
#     # 创建需要梯度的参数
#     theta_params = torch.randn(n_depth, n_qubits, requires_grad=True, dtype=DTYPE)
#
#     print(f"--- 1. 全局纠缠QNN (Strongly Entangling) ---")
#     print(f"输入: num_qubits={n_qubits}, num_depth={n_depth}")
#     print(f"参数 a 的形状: {theta_params.shape}\n")
#
#     # 构建全局QNN
#     global_qnn_matrix = build_global_qnn(n_qubits, n_depth, theta_params)
#
#     # 验证梯度
#     loss1 = torch.abs(global_qnn_matrix).sum()  # 构造一个标量损失
#     loss1.backward(retain_graph=True)  # retain_graph=True 以便为下一个网络复用 a
#
#     print(f"输出矩阵形状: {global_qnn_matrix.shape}")  # 应该是 16x16
#     print(f"参数 a 的梯度是否已计算: {theta_params.grad is not None}")
#     print("-" * 50)
#
#     # --- 2. 局部纠缠QNN (2-Local) ---
#     print(f"\n--- 2. 局部纠缠QNN (2-Local) ---")
#     # 清除之前的梯度
#     theta_params.grad.zero_()
#
#     # 构建局部QNN
#     local_qnn_matrix = build_local_qnn(n_qubits, n_depth, theta_params)
#
#     # 验证梯度
#     loss2 = torch.abs(local_qnn_matrix).sum()
#     loss2.backward()
#
#     print(f"输出矩阵形状: {local_qnn_matrix.shape}")  # 应该是 16x16
#     print(f"参数 a 的梯度是否已计算: {theta_params.grad is not None}")
#
#     # 最终断言，确保梯度确实存在
#     assert theta_params.grad is not None and torch.abs(theta_params.grad).sum() > 1e-9
