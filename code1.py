"""
教师异常行为预测
架构: 图像(MediaPipe + ST-GCN) + 结构化数据(Embedding + Attention) -> 多模态融合 -> 分类
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from PIL import Image
import os
import glob
import random
from collections import defaultdict
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ==================== 1. MediaPipe姿态估计器 ====================
class MediaPipePoseEstimator:
    """使用MediaPipe进行实时姿态估计"""

    def __init__(self, static_image_mode=False, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.num_joints = 33  # MediaPipe Pose有33个关键点

    def extract_keypoints(self, image):
        """
        从图像中提取人体关键点
        返回: (33, 3) 关键点坐标, 第三维为(x, y, visibility)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)

        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([
                    landmark.x,  # 归一化x坐标
                    landmark.y,  # 归一化y坐标
                    landmark.visibility  # 可见性分数
                ])
            return np.array(keypoints, dtype=np.float32)
        else:
            # 未检测到人，返回零数组
            return np.zeros((self.num_joints, 3), dtype=np.float32)

    def extract_sequence(self, image_frames):
        """从图像序列中提取关键点序列"""
        keypoint_sequence = []
        for frame in image_frames:
            keypoints = self.extract_keypoints(frame)
            keypoint_sequence.append(keypoints)

        # 转换为ST-GCN输入格式: (C, T, V, M)
        # C=3(x,y,visibility), T=帧数, V=关键点数, M=人数
        sequence = np.array(keypoint_sequence)  # (T, V, 3)
        sequence = sequence.transpose(2, 0, 1)  # (3, T, V)
        sequence = sequence[:, :, :, np.newaxis]  # (3, T, V, 1)
        return sequence

    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()


# ==================== 2. ST-GCN模型 ====================
class STGCNBlock(nn.Module):
    """ST-GCN基础块，包含正则化"""

    def __init__(self, in_channels, out_channels, stride=1, residual=True):
        super().__init__()

        # 空间图卷积
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 时间卷积
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1)  # 空间dropout
        )

        self.relu = nn.ReLU()
        self.residual = residual

        if not residual:
            self.residual_layer = lambda x: 0
        elif in_channels != out_channels or stride != 1:
            self.residual_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual_layer = nn.Identity()

    def forward(self, x):
        residual = self.residual_layer(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + residual)


class EnhancedSTGCN(nn.Module):
    """增强的ST-GCN模型，包含更多正则化和优化"""

    def __init__(self, in_channels=3, num_joints=33, dropout=0.3):
        super().__init__()

        # 数据标准化层
        self.normalize = nn.BatchNorm2d(in_channels)

        # 构建ST-GCN块
        self.stgcn_blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, residual=False),  # 第一个块不使用残差
            STGCNBlock(64, 64),
            STGCNBlock(64, 64),
            STGCNBlock(64, 128, stride=2),
            STGCNBlock(128, 128),
            STGCNBlock(128, 128),
            STGCNBlock(128, 256, stride=2),
            STGCNBlock(256, 256),
            STGCNBlock(256, 256)
        ])

        # 全局自适应池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 额外的正则化
        self.dropout = nn.Dropout(dropout)

        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, C, T, V)

        # 标准化输入
        x = self.normalize(x)

        # 通过ST-GCN块
        for block in self.stgcn_blocks:
            x = block(x)

        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # 特征投影和正则化
        x = self.feature_projection(x)

        return x


# ==================== 3. 结构化数据处理模块 ====================
class StructuredAttention(nn.Module):
    """双注意力机制处理结构化数据"""

    def __init__(self, input_dim, embedding_dim=50, dropout=0.3):
        super().__init__()

        # Layer Normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # 第一个Attention: 特征重加权
        self.attention1 = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()  # 输出0-1的权重
        )

        # 第二个Attention: 通道注意力
        self.attention2 = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

        # 残差连接的dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 第一个注意力: 特征级重加权
        residual = x
        x = self.norm1(x)
        weights1 = self.attention1(x)
        x = x * weights1  # 特征重加权

        # 第二个注意力: 通道级注意力
        x = self.norm2(x)
        weights2 = self.attention2(x)
        x = x * weights2

        # 残差连接
        x = residual + self.dropout(x)

        return x


class StructuredDataProcessor(nn.Module):
    """处理文本和数值特征的结构化数据处理器"""

    def __init__(self, vocab_size, num_numerical=7, embedding_dim=50, dropout=0.3):
        super().__init__()

        # 文本嵌入层（增加padding_idx处理未知词）
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 嵌入层后的标准化
        self.embed_norm = nn.LayerNorm(embedding_dim)

        # 数值特征的标准化
        self.numerical_norm = nn.BatchNorm1d(num_numerical)

        # 合并后的维度
        combined_dim = embedding_dim + num_numerical

        # 双注意力机制
        self.attention = StructuredAttention(combined_dim, dropout=dropout)

        # 特征投影层
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, text_indices, numerical_features):
        # text_indices: (B,)
        # numerical_features: (B, num_numerical)

        # 文本嵌入
        text_embedded = self.embedding(text_indices)  # (B, embedding_dim)
        text_embedded = self.embed_norm(text_embedded)

        # 数值特征标准化
        numerical = self.numerical_norm(numerical_features)

        # 拼接特征
        combined = torch.cat([text_embedded, numerical], dim=-1)  # (B, embedding_dim + num_numerical)

        # 应用双注意力
        attended = self.attention(combined)

        # 特征投影
        output = self.projection(attended)

        return output


# ==================== 4. 完整的多模态融合模型 ====================
class TeacherBehaviorPredictor(nn.Module):
    """完整的教师行为预测模型"""

    def __init__(self, vocab_size, num_classes=4, dropout_rate=0.3):
        super().__init__()

        self.dropout_rate = dropout_rate

        # ===== 视觉分支 =====
        self.visual_branch = nn.Sequential(
            EnhancedSTGCN(in_channels=3, num_joints=33, dropout=dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)  # DENSE Layer1
        )

        # ===== 结构化数据分支 =====
        self.structured_branch = StructuredDataProcessor(
            vocab_size=vocab_size,
            num_numerical=7,
            embedding_dim=50,
            dropout=dropout_rate
        )

        # ===== 融合分类头 =====
        self.fusion_classifier = nn.Sequential(
            # DENSE Layer3: 128+128=256 -> 64
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # DENSE Layer4: 64 -> 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # 输出层
            nn.Linear(32, num_classes)
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/Glorot初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, skeleton_seq, text_idx, numerical_features):
        # 视觉特征提取
        visual_features = self.visual_branch(skeleton_seq)  # (B, 128)

        # 结构化特征提取
        structured_features = self.structured_branch(text_idx, numerical_features)  # (B, 128)

        # 特征融合（直接相加）
        fused_features = visual_features + structured_features  # (B, 128)

        # 分类预测
        logits = self.fusion_classifier(fused_features)  # (B, num_classes)

        return logits

    def predict_proba(self, skeleton_seq, text_idx, numerical_features):
        """返回预测概率"""
        with torch.no_grad():
            logits = self.forward(skeleton_seq, text_idx, numerical_features)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def predict(self, skeleton_seq, text_idx, numerical_features):
        """返回预测类别（最大值）"""
        probabilities = self.predict_proba(skeleton_seq, text_idx, numerical_features)
        predictions = torch.argmax(probabilities, dim=-1)
        return predictions


# ==================== 5. 数据加载器 ====================
class TeacherBehaviorDataset(Dataset):
    """教师行为数据集"""

    def __init__(self, csv_path, image_base_dir, sequence_length=30,
                 transform=None, mode='train'):
        super().__init__()

        # 加载CSV数据
        self.df = pd.read_csv(csv_path)

        # 确保列数正确
        assert self.df.shape[1] == 9, f"CSV应有9列，实际有{self.df.shape[1]}列"

        # 图像基础路径
        self.image_base_dir = image_base_dir
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mode = mode

        # MediaPipe姿态估计器
        self.pose_estimator = MediaPipePoseEstimator(static_image_mode=True)

        # 缓存骨架序列以避免重复计算
        self.skeleton_cache = {}

        # 收集可用的图像文件夹
        self._collect_image_folders()

        print(f"数据集加载完成，共 {len(self.df)} 个样本")
        print(f"标签分布:\n{self.df.iloc[:, -1].value_counts().sort_index()}")

    def _collect_image_folders(self):
        """收集00和11文件夹中的所有图像文件夹"""
        self.image_folders = {'00': [], '11': []}

        for label_folder in ['00', '11']:
            folder_path = os.path.join(self.image_base_dir, label_folder)
            if os.path.exists(folder_path):
                subfolders = [f for f in os.listdir(folder_path)
                              if os.path.isdir(os.path.join(folder_path, f))]
                self.image_folders[label_folder] = sorted(subfolders, key=lambda x: int(x))

        print(f"找到 {len(self.image_folders['00'])} 个'00'文件夹")
        print(f"找到 {len(self.image_folders['11'])} 个'11'文件夹")

    def _load_image_sequence(self, label):
        """根据标签加载图像序列"""
        # 根据标签选择文件夹类型
        folder_type = '00' if label in [0, 1, 2] else '11'

        # 随机选择一个文件夹
        if not self.image_folders[folder_type]:
            raise ValueError(f"没有可用的 '{folder_type}' 文件夹")

        selected_folder = random.choice(self.image_folders[folder_type])
        folder_path = os.path.join(self.image_base_dir, folder_type, selected_folder)

        # 获取所有图像文件
        image_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')) +
                             glob.glob(os.path.join(folder_path, '*.png')))

        if not image_files:
            raise ValueError(f"文件夹 {folder_path} 中没有图像文件")

        # 随机选择起始点，加载连续帧
        if len(image_files) > self.sequence_length:
            start_idx = random.randint(0, len(image_files) - self.sequence_length)
            selected_files = image_files[start_idx:start_idx + self.sequence_length]
        else:
            # 如果图像不足，重复最后一帧
            selected_files = image_files
            while len(selected_files) < self.sequence_length:
                selected_files.append(image_files[-1])
            selected_files = selected_files[:self.sequence_length]

        # 加载图像
        images = []
        for img_file in selected_files:
            img = Image.open(img_file).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        return torch.stack(images)  # (T, C, H, W)

    def _extract_skeleton_sequence(self, image_sequence):
        """从图像序列提取骨架序列"""
        # 将张量转换回PIL图像用于MediaPipe
        images = []
        for img_tensor in image_sequence:
            # 反标准化并转换为PIL图像
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            images.append(img_pil)

        # 提取骨架序列
        skeleton_seq = self.pose_estimator.extract_sequence(images)
        return skeleton_seq

    def __getitem__(self, idx):
        # 获取CSV行数据
        row = self.df.iloc[idx]

        # 最后一列是标签
        label = int(row.iloc[-1])

        # 创建缓存键
        cache_key = f"{idx}_{label}"

        # 尝试从缓存获取骨架序列
        if cache_key in self.skeleton_cache:
            skeleton_seq = self.skeleton_cache[cache_key]
        else:
            # 加载图像序列
            image_sequence = self._load_image_sequence(label)

            # 提取骨架序列
            skeleton_seq = self._extract_skeleton_sequence(image_sequence)

            # 缓存结果
            if self.mode == 'train':  # 只在训练时缓存
                self.skeleton_cache[cache_key] = skeleton_seq

        # 结构化数据
        # 第2列是文本（索引1），转换为整数
        text_idx = int(row.iloc[1])

        # 其他数值特征（第0列和第3-8列，共7个特征）
        numerical_indices = [0] + list(range(3, 8))  # 索引: 0, 3, 4, 5, 6, 7, 8
        numerical_features = row.iloc[numerical_indices].astype(np.float32).values

        # 转换为张量
        skeleton_tensor = torch.FloatTensor(skeleton_seq).permute(3, 0, 1, 2)  # (1, 3, T, V)
        text_tensor = torch.LongTensor([text_idx])
        numerical_tensor = torch.FloatTensor(numerical_features)
        label_tensor = torch.LongTensor([label])

        return {
            'skeleton': skeleton_tensor.squeeze(0),  # (3, T, V)
            'text_idx': text_tensor.squeeze(),  # scalar
            'numerical': numerical_tensor,  # (7,)
            'label': label_tensor.squeeze()  # scalar
        }

    def __len__(self):
        return len(self.df)


# ==================== 6. 训练和验证函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        # 获取数据
        skeleton = batch['skeleton'].to(device)
        text_idx = batch['text_idx'].to(device)
        numerical = batch['numerical'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(skeleton, text_idx, numerical)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条
        accuracy = 100 * correct / total
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = 100 * correct / total

    return avg_loss, avg_accuracy


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Validation]')
        for batch in pbar:
            skeleton = batch['skeleton'].to(device)
            text_idx = batch['text_idx'].to(device)
            numerical = batch['numerical'].to(device)
            labels = batch['label'].to(device)

            outputs = model(skeleton, text_idx, numerical)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = 100 * correct / total

    return avg_loss, avg_accuracy, all_predictions, all_labels


# ==================== 7. 主训练流程 ====================
def main():
    # ===== 配置参数 =====
    config = {
        # 数据参数
        'csv_path': 'path/to/your/data.csv',
        'image_base_dir': 'path/to/image/folders',

        # 模型参数
        'vocab_size': 1000,  # 根据实际文本数据调整
        'num_classes': 4,
        'dropout_rate': 0.3,

        # 训练参数
        'batch_size': 8,  # 骨架序列处理需要较小batch
        'learning_rate': 0.001,
        'weight_decay': 1e-4,  # L2正则化
        'num_epochs': 50,
        'sequence_length': 30,

        # 数据划分
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,

        # 设备
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 60)
    print("教师异常行为预测模型训练")
    print("=" * 60)
    print(f"使用设备: {config['device']}")

    # ===== 准备数据 =====
    print("\n1. 准备数据集...")

    # 加载完整数据集
    full_dataset = TeacherBehaviorDataset(
        csv_path=config['csv_path'],
        image_base_dir=config['image_base_dir'],
        sequence_length=config['sequence_length'],
        mode='train'
    )

    # 划分数据集
    dataset_size = len(full_dataset)
    train_size = int(config['train_ratio'] * dataset_size)
    val_size = int(config['val_ratio'] * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    print(f"数据集划分: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if config['device'] == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )

    # ===== 初始化模型 =====
    print("\n2. 初始化模型...")
    model = TeacherBehaviorPredictor(
        vocab_size=config['vocab_size'],
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate']
    ).to(config['device'])

    # 打印模型架构
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ===== 损失函数和优化器 =====
    # 带标签平滑的交叉熵损失
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器（带权重衰减/L2正则化）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=5,
        factor=0.5,
        verbose=True
    )

    # ===== 训练循环 =====
    print("\n3. 开始训练...")
    best_val_acc = 0
    train_history = []

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'=' * 40}")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device'], epoch
        )

        # 验证
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, config['device']
        )

        # 更新学习率
        scheduler.step(val_acc)

        # 保存历史
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        print(f"\n训练结果:")
        print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_history': train_history,
                'config': config,
                'val_acc': val_acc,
            }, 'best_teacher_behavior_model.pth')
            print(f"  ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")

    # ===== 最终测试 =====
    print("\n4. 在测试集上评估...")
    model.load_state_dict(torch.load('best_teacher_behavior_model.pth')['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, config['device']
    )

    print(f"\n最终测试结果:")
    print(f"  测试损失: {test_loss:.4f}")
    print(f"  测试准确率: {test_acc:.2f}%")

    # ===== 保存完整训练记录 =====
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_history': train_history,
        'test_results': {
            'loss': test_loss,
            'accuracy': test_acc,
            'predictions': test_preds,
            'labels': test_labels
        },
        'vocab_size': config['vocab_size']
    }

    torch.save(final_checkpoint, 'final_teacher_behavior_model.pth')
    print("\n✓ 训练完成！模型已保存为 'final_teacher_behavior_model.pth'")


# ==================== 8. 推理示例 ====================
def predict_single_sample(model_path, csv_row, image_folder_base):
    """对单个样本进行预测"""
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']

    model = TeacherBehaviorPredictor(
        vocab_size=checkpoint.get('vocab_size', config['vocab_size']),
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 准备数据
    label = int(csv_row.iloc[-1])

    # 创建临时数据集对象
    from torch.utils.data import Dataset as TorchDataset
    class SingleSampleDataset(TorchDataset):
        def __init__(self):
            self.pose_estimator = MediaPipePoseEstimator(static_image_mode=True)
            self.df = pd.DataFrame([csv_row])
            self.image_folders = {'00': ['0'], '11': ['0']}  # 简化
            self.sequence_length = config['sequence_length']

        def __len__(self): return 1

        def __getitem__(self, idx):
            # 简化的数据加载逻辑
            return {
                'skeleton': torch.randn(3, 30, 33),  # 实际应加载真实数据
                'text_idx': torch.LongTensor([int(csv_row.iloc[1])]),
                'numerical': torch.FloatTensor([csv_row.iloc[0]] + list(csv_row.iloc[3:8])),
                'label': torch.LongTensor([label])
            }

    dataset = SingleSampleDataset()
    sample = dataset[0]

    # 预测
    with torch.no_grad():
        skeleton_seq = sample['skeleton'].unsqueeze(0)
        text_idx = sample['text_idx'].unsqueeze(0)
        numerical = sample['numerical'].unsqueeze(0)

        logits = model(skeleton_seq, text_idx, numerical)
        probabilities = F.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()

    return {
        'predicted_class': prediction,
        'probabilities': probabilities.squeeze().numpy(),
        'confidence': probabilities.max().item()
    }


# ==================== 9. 运行主程序 ====================
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 运行训练
    main()

    print("\n" + "=" * 60)
    print("使用说明:")
    print("1. 修改config中的csv_path和image_base_dir为您的数据路径")
    print("2. 根据实际文本数据调整vocab_size参数")
    print("3. 模型支持4分类任务（标签0-3）")
    print("4. 训练好的模型可以用于预测新样本")
    print("=" * 60)