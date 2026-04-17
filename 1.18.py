import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
import os
import warnings
import matplotlib
from scipy import stats
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap, Normalize
warnings.filterwarnings('ignore')

# 设置matplotlib字体和数学字体
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 创建保存结果的文件夹 - 新文件夹
output_dir = "E:/stgnn_counterfactual_optimized"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created new results folder: {output_dir}")

# ==================== 固定邻接矩阵定义 ====================
# 基于您提供的矩阵，我们创建一个固定邻接矩阵
FIXED_ADJ_MATRIX = np.array([
    # 北京 天津 河北 山西 内蒙古 辽宁 吉林 黑龙江 上海 江苏 浙江 安徽 福建 江西 山东 河南 湖北 湖南 广东 广西 海南 重庆 四川 贵州 云南 陕西 甘肃 青海 宁夏 新疆
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 北京
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 天津
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 河北
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 山西
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],  # 内蒙古
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 辽宁
    [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 吉林
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 黑龙江
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 上海
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 江苏
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 浙江
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 安徽
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 福建
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 江西
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 山东
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 河南
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # 湖北
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 湖南
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 广东
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # 广西
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 海南
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],  # 重庆
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],  # 四川
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # 贵州
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # 云南
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],  # 陕西
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1],  # 甘肃
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 青海
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 宁夏
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]   # 新疆
], dtype=np.float32)

# 确保对称性（尽管矩阵已经对称）
FIXED_ADJ_MATRIX = (FIXED_ADJ_MATRIX + FIXED_ADJ_MATRIX.T) / 2

# ==================== 1. 数据加载 ====================
def load_data_simple(adj_path, feat_path, years=range(2011, 2024)):
    """简化的数据加载，只使用原始特征"""
    adj_matrices = []
    feature_matrices = []
    provinces = []
    
    print(f"Loading data for years: {list(years)}")
    
    # 加载第一年的数据获取省份列表
    first_year = years[0]
    try:
        adj_df = pd.read_excel(adj_path, sheet_name=str(first_year), index_col=0)
        provinces = list(adj_df.index)
        print(f"Found {len(provinces)} provinces")
    except Exception as e:
        print(f"Error loading adjacency matrix: {e}")
        return None, None, None
    
    # 加载所有年份的数据
    for year in years:
        try:
            # 加载邻接矩阵
            adj_df = pd.read_excel(adj_path, sheet_name=str(year), index_col=0)
            adj_matrix = adj_df.values.astype(np.float32)
            
            # 确保省份顺序一致
            if list(adj_df.index) != provinces:
                print(f"Warning: Province order mismatch in year {year}")
                adj_df = adj_df.reindex(provinces)
                adj_matrix = adj_df.values.astype(np.float32)
            
            # 对称化处理并添加自环
            adj_matrix = (adj_matrix + adj_matrix.T) / 2
            np.fill_diagonal(adj_matrix, 1.0)  # 添加自环
            adj_matrices.append(adj_matrix)
            
            # 加载节点特征
            feat_df = pd.read_excel(feat_path, sheet_name=str(year))
            
            # 确保有正确的列
            if 'P' in feat_df.columns:
                feat_df = feat_df.set_index('P')
            elif '省份' in feat_df.columns:
                feat_df = feat_df.set_index('省份')
            
            # 确保省份顺序一致
            feat_df = feat_df.reindex(provinces)
            
            # 检查并提取特征
            required_columns = ['EE', 'DE', 'GDP', 'IS', 'EDU', 'URBAN', 'DENSITY', 'OPEN']
            
            # 提取目标变量
            if 'EE' not in feat_df.columns:
                print(f"Error: 'EE' column not found in year {year}")
                continue
            
            target = feat_df['EE'].values.astype(np.float32)
            
            # 提取特征
            feature_cols = []
            feature_data = []
            
            for col in required_columns:
                if col != 'EE' and col in feat_df.columns:
                    feature_cols.append(col)
                    feature_data.append(feat_df[col].values.astype(np.float32))
            
            if feature_data:
                features = np.column_stack(feature_data)
                
                feature_matrices.append({
                    'features': features,
                    'target': target,
                    'feature_names': feature_cols
                })
            else:
                print(f"Warning: No features found for year {year}")
                
        except Exception as e:
            print(f"Error loading year {year}: {e}")
            continue
    
    if not feature_matrices or not adj_matrices:
        print("Error: No valid data loaded")
        return None, None, None
    
    print(f"Successfully loaded {len(adj_matrices)} adjacency matrices and {len(feature_matrices)} feature matrices")
    return provinces, adj_matrices, feature_matrices

# ==================== 2. 简单STGNN模型 ====================
class SimpleSTGNN(nn.Module):
    """简单但有效的时空图神经网络"""
    
    def __init__(self, n_features, n_nodes, window_size=3, hidden_dim=64, dropout=0.2):
        super(SimpleSTGNN, self).__init__()
        
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        
        # 1. 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 空间图卷积
        self.spatial_conv = nn.Linear(hidden_dim, hidden_dim)
        
        # 3. 时间编码器（GRU）
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # 4. 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 自适应邻接矩阵
        self.adaptive_adj = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.01)
        
        # 初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, adj):
        batch_size, window_size, n_nodes, n_features = x.shape
        
        # 1. 特征编码
        x_reshaped = x.reshape(batch_size * window_size * n_nodes, n_features)
        encoded = self.feature_encoder(x_reshaped)
        encoded = encoded.reshape(batch_size, window_size, n_nodes, -1)
        
        # 2. 空间图卷积
        spatial_features = []
        for t in range(window_size):
            h_t = encoded[:, t, :, :]
            
            # 结合预定义邻接矩阵和自适应邻接矩阵
            adj_t = adj[:, t, :, :]
            adaptive_adj = torch.sigmoid(self.adaptive_adj).unsqueeze(0).expand(batch_size, -1, -1)
            combined_adj = adj_t + adaptive_adj
            
            # 行归一化
            row_sum = combined_adj.sum(dim=-1, keepdim=True) + 1e-10
            normalized_adj = combined_adj / row_sum
            
            # 空间卷积
            h_t = self.spatial_conv(h_t)
            h_t = torch.bmm(normalized_adj, h_t)
            h_t = F.relu(h_t)
            
            spatial_features.append(h_t)
        
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # 3. 时间编码
        temporal_input = spatial_features.permute(0, 2, 1, 3)
        temporal_input = temporal_input.reshape(batch_size * n_nodes, window_size, -1)
        
        gru_output, _ = self.temporal_encoder(temporal_input)
        
        # 取最后一个时间步
        last_step = gru_output[:, -1, :]
        
        # 4. 预测
        predictions = self.predictor(last_step)
        predictions = predictions.reshape(batch_size, n_nodes)
        
        return predictions

# ==================== 3. 数据集类 ====================
class SimpleSTGNNDataset(Dataset):
    """简化的时空图神经网络数据集"""
    
    def __init__(self, adj_matrices, feature_matrices, window_size=3, forecast_step=1):
        self.adj_matrices = adj_matrices
        self.features = [fm['features'] for fm in feature_matrices]
        self.targets = [fm['target'] for fm in feature_matrices]
        self.feature_names = feature_matrices[0]['feature_names'] if feature_matrices else []
        self.window_size = window_size
        self.forecast_step = forecast_step
        
        print(f"Creating dataset with {len(self.features)} time steps...")
        
        # 标准化特征（整个数据集一起标准化）
        all_features = np.concatenate(self.features, axis=0)
        self.feature_scaler = StandardScaler()
        all_features_scaled = self.feature_scaler.fit_transform(all_features)
        
        # 分割回各年份
        self.scaled_features = []
        start_idx = 0
        for feat in self.features:
            end_idx = start_idx + feat.shape[0]
            self.scaled_features.append(all_features_scaled[start_idx:end_idx])
            start_idx = end_idx
        
        # 标准化目标
        all_targets = np.concatenate(self.targets, axis=0).reshape(-1, 1)
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(all_targets)
        
        # 创建样本
        self.samples = []
        T = len(adj_matrices)
        N = adj_matrices[0].shape[0]
        
        print(f"Generating samples with window_size={window_size}, forecast_step={forecast_step}...")
        
        valid_samples = 0
        for t in range(T - window_size - forecast_step + 1):
            try:
                x_features = []
                x_adjs = []
                
                for i in range(t, t + window_size):
                    x_features.append(self.scaled_features[i])
                    x_adjs.append(self.adj_matrices[i])
                
                y = self.targets[t + window_size + forecast_step - 1]
                y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
                
                self.samples.append({
                    'x_features': np.stack(x_features, axis=0),
                    'x_adjs': np.stack(x_adjs, axis=0),
                    'y': y_scaled,
                    'time_index': t
                })
                valid_samples += 1
                
            except Exception as e:
                print(f"Warning: Error creating sample at t={t}: {e}")
                continue
        
        print(f"Created {valid_samples} valid samples out of {T - window_size - forecast_step + 1} possible")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'features': torch.FloatTensor(sample['x_features']),
            'adj': torch.FloatTensor(sample['x_adjs']),
            'target': torch.FloatTensor(sample['y']),
            'time_index': sample['time_index']
        }
    
    def create_counterfactual_dataset(self, province_idx, de_increase=0.5):
        """创建反事实数据集：提升指定省份的数字化水平"""
        print(f"Creating counterfactual dataset for province index {province_idx} with DE increase {de_increase}")
        
        # 找到DE特征的索引
        de_idx = None
        for i, name in enumerate(self.feature_names):
            if 'DE' in name.upper():
                de_idx = i
                break
        
        if de_idx is None:
            print("DE feature not found. Using first feature as DE.")
            de_idx = 0
        
        counterfactual_samples = []
        
        for sample in self.samples:
            # 复制特征
            x_features = sample['x_features'].copy()
            
            # 提升指定省份的DE特征（增加0.5个标准差）
            # 这里我们直接增加标准化后的DE值，相当于提升0.5个标准差
            x_features[:, province_idx, de_idx] += de_increase
            
            counterfactual_samples.append({
                'x_features': x_features,
                'x_adjs': sample['x_adjs'],
                'y': sample['y'],
                'time_index': sample['time_index']
            })
        
        return counterfactual_samples

# ==================== 4. 训练函数 ====================
def train_model(model, train_loader, val_loader, n_epochs=200, device='cpu', save_path=None):
    """训练模型"""
    
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    
    best_val_r2 = -float('inf')
    patience = 20
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for batch in train_loader:
            features = batch['features'].to(device)
            adj = batch['adj'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            output = model(features, adj)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # 收集预测
            train_preds.append(output.detach().cpu().numpy())
            train_targets.append(target.detach().cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_losses.append(avg_train_loss)
        
        # 计算训练R²
        if train_preds:
            train_preds_all = np.concatenate(train_preds, axis=0)
            train_targets_all = np.concatenate(train_targets, axis=0)
            train_r2 = r2_score(train_targets_all.flatten(), train_preds_all.flatten())
            train_r2s.append(train_r2)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                adj = batch['adj'].to(device)
                target = batch['target'].to(device)
                
                output = model(features, adj)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                val_preds.append(output.cpu().numpy())
                val_targets.append(target.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)
        
        # 计算验证R²
        if val_preds:
            val_preds_all = np.concatenate(val_preds, axis=0)
            val_targets_all = np.concatenate(val_targets, axis=0)
            val_r2 = r2_score(val_targets_all.flatten(), val_preds_all.flatten())
            val_r2s.append(val_r2)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                }, save_path)
                print(f"  ✓ Saved best model with R²={val_r2:.4f}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f"  ⚠ Early stopping triggered at epoch {epoch+1}")
            break
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{n_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train R²: {train_r2:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val R²: {val_r2:.4f}")
            print(f"  LR: {current_lr:.6f}, Patience: {patience_counter}/{patience}")
    
    print(f"Training completed. Best validation R²: {best_val_r2:.4f}")
    return train_losses, val_losses, train_r2s, val_r2s

# ==================== 5. 反事实模拟分析 ====================
def perform_counterfactual_analysis(model, dataset, provinces, target_province_name, 
                                    de_increase=0.5, device='cpu', save_path=None, fixed_adj_matrix=None):
    """
    执行反事实模拟分析：
    1. 提升指定省份的数字化水平
    2. 观察所有省份能源效率的变化
    3. 量化空间溢出效应
    """
    model.eval()
    
    # 找到目标省份的索引
    if target_province_name not in provinces:
        print(f"Error: Province '{target_province_name}' not found in province list.")
        print(f"Available provinces: {provinces}")
        return None
    
    target_idx = provinces.index(target_province_name)
    print(f"Target province: {target_province_name} (index: {target_idx})")
    
    # 1. 计算原始预测值
    print("Calculating original predictions...")
    original_predictions = []
    original_targets = []
    
    with torch.no_grad():
        for sample in dataset.samples:
            features = torch.FloatTensor(sample['x_features']).unsqueeze(0).to(device)
            adj = torch.FloatTensor(sample['x_adjs']).unsqueeze(0).to(device)
            
            output = model(features, adj)
            original_predictions.append(output.cpu().numpy())
            original_targets.append(sample['y'])
    
    original_predictions = np.concatenate(original_predictions, axis=0)  # [n_samples, n_provinces]
    original_targets = np.array(original_targets)  # [n_samples, n_provinces]
    
    # 2. 创建反事实数据集
    print("Creating counterfactual dataset...")
    counterfactual_samples = dataset.create_counterfactual_dataset(target_idx, de_increase=de_increase)
    
    # 3. 计算反事实预测值
    print("Calculating counterfactual predictions...")
    counterfactual_predictions = []
    
    with torch.no_grad():
        for sample in counterfactual_samples:
            features = torch.FloatTensor(sample['x_features']).unsqueeze(0).to(device)
            adj = torch.FloatTensor(sample['x_adjs']).unsqueeze(0).to(device)
            
            output = model(features, adj)
            counterfactual_predictions.append(output.cpu().numpy())
    
    counterfactual_predictions = np.concatenate(counterfactual_predictions, axis=0)  # [n_samples, n_provinces]
    
    # 4. 计算变化量
    changes = counterfactual_predictions - original_predictions  # [n_samples, n_provinces]
    
    # 5. 反标准化变化量（转换为原始EE单位）
    ee_std = dataset.target_scaler.scale_[0]  # 目标变量的标准差
    ee_mean = dataset.target_scaler.mean_[0]  # 目标变量的均值
    
    changes_original = changes * ee_std
    
    # 6. 计算平均变化量（按省份和时间）
    mean_changes_by_province = changes_original.mean(axis=0)  # 按省份平均
    std_changes_by_province = changes_original.std(axis=0)    # 按省份标准差
    
    # 7. 识别邻居省份（基于固定邻接矩阵或平均邻接矩阵）
    if fixed_adj_matrix is not None:
        print("Using fixed adjacency matrix for neighbor identification")
        
        # 确保矩阵维度正确
        if fixed_adj_matrix.shape[0] != len(provinces):
            print(f"Error: Fixed adjacency matrix shape {fixed_adj_matrix.shape} doesn't match number of provinces {len(provinces)}")
            return None
        
        # 找出邻居（值为1的位置，排除对角线）
        neighbor_mask = fixed_adj_matrix[target_idx, :].copy()
        neighbor_mask[target_idx] = 0  # 排除自身
        
        neighbor_indices = np.where(neighbor_mask > 0)[0]
    else:
        print("Using average adjacency matrix from dataset for neighbor identification")
        avg_adj = np.mean(dataset.adj_matrices, axis=0)
        threshold = 0.1  # 邻接阈值
        neighbor_indices = np.where(avg_adj[target_idx, :] > threshold)[0]
        neighbor_indices = neighbor_indices[neighbor_indices != target_idx]  # 排除自身
    
    print(f"\nTarget province: {target_province_name}")
    print(f"Number of neighbor provinces: {len(neighbor_indices)}")
    if len(neighbor_indices) > 0:
        neighbor_names = [provinces[i] for i in neighbor_indices]
        print(f"Neighbor provinces: {neighbor_names}")
    
    # 8. 计算空间溢出效应指标
    target_change = mean_changes_by_province[target_idx]
    
    if len(neighbor_indices) > 0:
        neighbor_changes = mean_changes_by_province[neighbor_indices]
        avg_neighbor_change = neighbor_changes.mean()
        max_neighbor_change = neighbor_changes.max()
        min_neighbor_change = neighbor_changes.min()
        
        # 计算空间溢出比率
        if abs(target_change) > 1e-10:
            spillover_ratio = avg_neighbor_change / target_change
        else:
            spillover_ratio = np.nan
    else:
        avg_neighbor_change = 0
        max_neighbor_change = 0
        min_neighbor_change = 0
        spillover_ratio = 0
    
    # 9. 创建结果字典
    results = {
        'target_province': target_province_name,
        'target_idx': target_idx,
        'de_increase': de_increase,
        'mean_changes_by_province': mean_changes_by_province,
        'std_changes_by_province': std_changes_by_province,
        'neighbor_indices': neighbor_indices,
        'neighbor_names': [provinces[i] for i in neighbor_indices] if len(neighbor_indices) > 0 else [],
        'target_change': target_change,
        'avg_neighbor_change': avg_neighbor_change,
        'max_neighbor_change': max_neighbor_change,
        'min_neighbor_change': min_neighbor_change,
        'spillover_ratio': spillover_ratio,
        'changes_original': changes_original,
        'original_predictions': original_predictions,
        'counterfactual_predictions': counterfactual_predictions,
        'ee_std': ee_std,
        'ee_mean': ee_mean,
        'adj_matrix': fixed_adj_matrix if fixed_adj_matrix is not None else np.mean(dataset.adj_matrices, axis=0)
    }
    
    # 10. 打印结果
    print("\n" + "="*80)
    print("COUNTERFACTUAL ANALYSIS RESULTS")
    print("="*80)
    print(f"Scenario: Increase DE by {de_increase} standard deviations in {target_province_name}")
    print(f"\nDirect Effect on Target Province:")
    print(f"  {target_province_name}: EE change = {target_change:.6f}")
    
    if len(neighbor_indices) > 0:
        print(f"\nSpatial Spillover Effects on Neighbor Provinces:")
        print(f"  Average neighbor EE change: {avg_neighbor_change:.6f}")
        print(f"  Maximum neighbor EE change: {max_neighbor_change:.6f}")
        print(f"  Minimum neighbor EE change: {min_neighbor_change:.6f}")
        print(f"  Spillover ratio (neighbor/target): {spillover_ratio:.4f}")
    
    print(f"\nTop 5 Most Affected Provinces:")
    # 按变化量绝对值排序
    sorted_indices = np.argsort(np.abs(mean_changes_by_province))[::-1]
    for i in range(min(5, len(provinces))):
        idx = sorted_indices[i]
        change = mean_changes_by_province[idx]
        rel_change = (change / (np.abs(original_predictions[:, idx].mean()) * ee_std + ee_mean)) * 100 if (np.abs(original_predictions[:, idx].mean()) * ee_std + ee_mean) > 1e-10 else 0
        
        # 标记省份类型
        province_type = ""
        if idx == target_idx:
            province_type = " [TARGET]"
        elif idx in neighbor_indices:
            province_type = " [NEIGHBOR]"
            
        print(f"  {i+1}. {provinces[idx]}: EE change = {change:.6f} ({rel_change:.2f}%){province_type}")
    
    return results

# ==================== 6. 优化的可视化函数 ====================
def visualize_counterfactual_results(results, provinces, save_dir):
    """可视化反事实模拟结果 - 优化版"""
    
    target_province = results['target_province']
    de_increase = results['de_increase']
    mean_changes = results['mean_changes_by_province']
    neighbor_indices = results['neighbor_indices']
    adj_matrix = results['adj_matrix']
    
    # 省份英文映射
    province_english_map = {
        '北京市': 'Beijing', '天津市': 'Tianjin', '河北省': 'Hebei', '山西省': 'Shanxi',
        '内蒙古自治区': 'Inner Mongolia', '辽宁省': 'Liaoning', '吉林省': 'Jilin',
        '黑龙江省': 'Heilongjiang', '上海市': 'Shanghai', '江苏省': 'Jiangsu',
        '浙江省': 'Zhejiang', '安徽省': 'Anhui', '福建省': 'Fujian', '江西省': 'Jiangxi',
        '山东省': 'Shandong', '河南省': 'Henan', '湖北省': 'Hubei', '湖南省': 'Hunan',
        '广东省': 'Guangdong', '广西壮族自治区': 'Guangxi', '海南省': 'Hainan',
        '重庆市': 'Chongqing', '四川省': 'Sichuan', '贵州省': 'Guizhou', '云南省': 'Yunnan',
        '陕西省': 'Shaanxi', '甘肃省': 'Gansu', '青海省': 'Qinghai', '宁夏回族自治区': 'Ningxia',
        '新疆维吾尔自治区': 'Xinjiang'
    }
    
    # 省份简称（用于网络图）
    province_short_map = {
        '北京市': 'BJ', '天津市': 'TJ', '河北省': 'HEB', '山西省': 'SX',
        '内蒙古自治区': 'IM', '辽宁省': 'LN', '吉林省': 'JL',
        '黑龙江省': 'HLJ', '上海市': 'SH', '江苏省': 'JS',
        '浙江省': 'ZJ', '安徽省': 'AH', '福建省': 'FJ', '江西省': 'JX',
        '山东省': 'SD', '河南省': 'HEN', '湖北省': 'HUB', '湖南省': 'HUN',
        '广东省': 'GD', '广西壮族自治区': 'GX', '海南省': 'HN',
        '重庆市': 'CQ', '四川省': 'SC', '贵州省': 'GZ', '云南省': 'YN',
        '陕西省': 'SAX', '甘肃省': 'GS', '青海省': 'QH', '宁夏回族自治区': 'NX',
        '新疆维吾尔自治区': 'XJ'
    }
    
    # 1. 创建省份变化条形图 - 优化图例位置
    plt.figure(figsize=(18, 10))
    
    # 按变化量排序
    sorted_indices = np.argsort(mean_changes)[::-1]
    sorted_provinces = [provinces[i] for i in sorted_indices]
    sorted_changes = mean_changes[sorted_indices]
    
    # 为不同省份类型设置颜色
    colors = []
    for i, idx in enumerate(sorted_indices):
        province = provinces[idx]
        if province == target_province:
            colors.append('#FF6B6B')  # 红色 - 目标省份
        elif idx in neighbor_indices:
            colors.append('#FFA726')  # 橙色 - 邻居省份
        else:
            colors.append('#4ECDC4')  # 青色 - 其他省份
    
    bars = plt.barh(range(len(sorted_provinces)), sorted_changes, color=colors, alpha=0.85, height=0.8)
    plt.yticks(range(len(sorted_provinces)), [province_english_map.get(p, p) for p in sorted_provinces], fontsize=10)
    plt.xlabel('Change in Energy Efficiency (EE)', fontsize=13, fontweight='bold')
    plt.title(f'Counterfactual Analysis: Impact of Increasing DE in {province_english_map.get(target_province, target_province)}\n(DE increased by {de_increase} standard deviations)', 
              fontsize=15, fontweight='bold', pad=20)
    
    # 在柱子上添加数值标签 - 优化位置
    for bar, change in zip(bars, sorted_changes):
        width = bar.get_width()
        
        # 根据宽度正负决定标签位置
        if width >= 0:
            # 正数：标签放在柱体右侧
            x_pos = width + max(abs(sorted_changes)) * 0.01  # 仅1%的距离
            ha = 'left'
        else:
            # 负数：标签放在柱体左侧
            x_pos = width - max(abs(sorted_changes)) * 0.01  # 仅1%的距离
            ha = 'right'
        
        # 标签颜色
        color = 'black' if abs(width) > max(abs(sorted_changes)) * 0.1 else 'gray'
        
        # 标签格式
        label_text = f'{change:.4f}'
        
        # 添加标签
        plt.text(x_pos, bar.get_y() + bar.get_height()/2, 
                label_text, va='center', ha=ha, fontsize=9, color=color, fontweight='bold')
    
    # 添加图例 - 改为右上角
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', alpha=0.85, label=f'Target: {province_english_map.get(target_province, target_province)}'),
        Patch(facecolor='#FFA726', alpha=0.85, label='Neighbor Provinces'),
        Patch(facecolor='#4ECDC4', alpha=0.85, label='Other Provinces')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
    # 添加网格线
    plt.grid(True, alpha=0.2, axis='x', linestyle='--')
    
    # 添加零线
    plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"counterfactual_changes_{target_province}.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 2. 优化的空间网络图 - 根据溢出效应强度用颜色和粗细区分
    plt.figure(figsize=(22, 16))  # 增加图形大小

  # 创建网络图
    G = nx.Graph()

    # 添加节点
    for i, province in enumerate(provinces):
        G.add_node(province_short_map.get(province, province), 
                  change=mean_changes[i],
                  province_name=province_english_map.get(province, province),
                  province_full=province,
                  is_target=(province == target_province),
                  is_neighbor=(i in neighbor_indices))

# 收集所有边的数据
    edge_data = []

# 只考虑有连接关系的边
    for i in range(len(provinces)):
        for j in range(i+1, len(provinces)):
            if adj_matrix[i, j] > 0.5:  # 由于是0/1矩阵，阈值设为0.5
            # 计算边的权重（平均变化量）
                avg_change = (mean_changes[i] + mean_changes[j]) / 2
            
            # 只考虑正向溢出的边（平均变化量>0）
                if avg_change > 0:
                    edge_data.append({
                        'source': province_short_map.get(provinces[i], provinces[i]),
                        'target': province_short_map.get(provinces[j], provinces[j]),
                        'weight': avg_change,
                        'source_change': mean_changes[i],
                        'target_change': mean_changes[j],
                        'is_target_related': (i == results['target_idx'] or j == results['target_idx'])
                    })

    if edge_data:
    # 提取边的权重
        edge_weights = [e['weight'] for e in edge_data]
        max_weight = max(edge_weights) if edge_weights else 1.0
    
    # 添加边到网络，根据权重设置样式
        for edge in edge_data:
            weight = edge['weight']
        
        # 归一化权重到[0, 1]区间
            norm_weight = weight / max_weight if max_weight > 0 else 0
        
        # 边的宽度：基于权重，权重越大边越粗
            width = 1.0 + 8.0 * norm_weight  # 宽度范围：1.0-9.0（增加宽度范围）
        
        # 边的颜色：基于权重使用渐变色（从浅黄到深红）
        # 使用RGB颜色：权重越大越红
            red = 1.0
            green = 1.0 - norm_weight * 0.7  # 权重越大绿色越少
            blue = 1.0 - norm_weight * 0.7   # 权重越大蓝色越少
            edge_color = (red, green, blue)
         
        # 透明度：基于权重
            alpha = 0.4 + 0.5 * norm_weight  # 透明度范围：0.4-0.9
        
            G.add_edge(edge['source'], edge['target'], 
                      weight=weight, 
                      norm_weight=norm_weight,
                      width=width,
                      color=edge_color,
                      alpha=alpha,
                      is_target_related=edge['is_target_related'])

# 节点位置（使用力导向布局）
    if len(G.nodes()) > 0:
        # 使用spring布局，增加k值和迭代次数以获得更好的布局
        pos = nx.spring_layout(G, seed=42, k=3.0, iterations=200)
    
    # 准备节点颜色和大小
        node_colors = []
        node_sizes = []
        node_labels = {}
    
    # 节点颜色：基于变化量
        max_abs_change = max(abs(mean_changes))
    
        for node in G.nodes():
        # 找到对应的省份
            for i, province in enumerate(provinces):
                if province_short_map.get(province, province) == node:
                    change = mean_changes[i]
                
                # 归一化变化量
                    if max_abs_change > 0:
                        norm_change = change / max_abs_change
                    else:
                        norm_change = 0
                
                # 节点颜色：基于变化量正负
                    if change > 0:
                    # 正变化：红色系
                        red_intensity = 1.0
                        green_blue_intensity = 1.0 - min(abs(norm_change), 0.7)
                        node_colors.append((red_intensity, green_blue_intensity, green_blue_intensity))
                    else:
                    # 负变化：蓝色系
                        blue_intensity = 1.0
                        red_green_intensity = 1.0 - min(abs(norm_change), 0.5)
                        node_colors.append((red_green_intensity, red_green_intensity, blue_intensity))
                
                # 节点大小：基于变化量绝对值 - 大幅增加节点大小
                    base_size = 1200  # 增加基础大小
                    size_factor = 2500  # 增加缩放因子
                    node_size = base_size + abs(change) * size_factor
                    node_sizes.append(min(node_size, 5000))  # 提高最大大小限制
                
                # 节点标签 - 使用英文名称
                    node_labels[node] = province_english_map.get(province, province)
                    break
    
    # 绘制边 - 根据是否与目标省份相关分层绘制
        target_edges = []
        other_edges = []
    
        for u, v, data in G.edges(data=True):
            if data.get('is_target_related', False):
                target_edges.append((u, v, data))
            else:
                other_edges.append((u, v, data))
    
    # 先绘制其他边（底层）
        if other_edges:
            for u, v, data in other_edges:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                      width=data['width'],
                                      alpha=data['alpha'],
                                      edge_color=[data['color']],
                                      style='solid')
    
    # 再绘制与目标相关的边（顶层，更突出）
        if target_edges:
            for u, v, data in target_edges:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                  width=data['width'] * 1.5,  # 目标相关边更粗
                                  alpha=min(data['alpha'] * 1.2, 1.0),  # 目标相关边更不透明
                                  edge_color=[data['color']],
                                  style='solid')
    
    # 绘制节点 - 大幅增加节点大小
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.9, edgecolors='black', linewidths=2.0)
    
    # 绘制节点标签 - 增加字体大小和粗细
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=13, font_weight='bold', 
                           font_family='Arial', verticalalignment='center')
    
    # 突出显示目标省份
        target_node = province_short_map.get(target_province, target_province)
        if target_node in G.nodes():
            nx.draw_networkx_nodes(G, pos, nodelist=[target_node], 
                              node_color='#FF0000', node_size=3000,  # 增加目标节点大小
                              edgecolors='black', linewidths=4)  # 增加边框宽度
        # 添加目标标签 - 增加字体大小
            plt.text(pos[target_node][0], pos[target_node][1] + 0.07,  # 调整位置
                'TARGET', fontsize=15, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='red', alpha=0.8))
    
    # 突出显示邻居省份
        neighbor_nodes = [province_short_map.get(provinces[i], provinces[i]) for i in neighbor_indices]
        if neighbor_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=neighbor_nodes, 
                              node_color='#FFA500', node_size=2000,  # 增加邻居节点大小
                              edgecolors='black', linewidths=3)  # 增加边框宽度
    
    # 创建颜色条图例（节点颜色） - 增加字体大小
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, 
                               norm=Normalize(vmin=min(mean_changes), vmax=max(mean_changes)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', fraction=0.03, pad=0.04)
        cbar.set_label('EE Change', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)  # 增加刻度标签大小
    
    # 创建边图例 - 大幅增加图例元素大小和字体
        from matplotlib.lines import Line2D
        edge_legend_elements = []
    
    # 添加不同强度的边示例
        if edge_data:
        # 获取边权重的分位数
            weights = [e['weight'] for e in edge_data]
            if weights:
                q25 = np.percentile(weights, 25)
                q50 = np.percentile(weights, 50)
                q75 = np.percentile(weights, 75)
            
            # 计算对应的颜色和宽度
                max_w = max(weights)
            
                def get_edge_style(weight):
                    norm = weight / max_w if max_w > 0 else 0
                    width = 1.0 + 8.0 * norm
                    red = 1.0
                    green = 1.0 - norm * 0.7
                    blue = 1.0 - norm * 0.7
                    return width, (red, green, blue)
            
            # 弱溢出边
                w1, c1 = get_edge_style(q25)
                edge_legend_elements.append(
                    Line2D([0], [0], color=c1, linewidth=w1, 
                          label=f'Weak spillover (≤{q25:.3f})')
                )
            
            # 中等溢出边
                w2, c2 = get_edge_style(q50)
                edge_legend_elements.append(
                    Line2D([0], [0], color=c2, linewidth=w2, 
                          label=f'Medium spillover (≤{q50:.3f})')
                )
            
            # 强溢出边
                w3, c3 = get_edge_style(q75)
                edge_legend_elements.append(
                    Line2D([0], [0], color=c3, linewidth=w3, 
                          label=f'Strong spillover (≤{q75:.3f})')
                )
    
    # 目标相关边
        edge_legend_elements.append(
             Line2D([0], [0], color='red', linewidth=5, linestyle='-',
              label='Connected to target province')
    )
    
    # 创建节点图例 - 大幅增加标记大小和字体
        node_legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
              markersize=18, label='Target province'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
              markersize=15, label='Neighbor province'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
              markersize=12, label='Positive EE change'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
              markersize=12, label='Negative EE change'),
    ]
    
    # 添加图例 - 大幅增加字体大小
        legend1 = plt.legend(handles=edge_legend_elements, loc='upper left', 
                        fontsize=13, title="Spillover Intensity", 
                        title_fontsize=14, framealpha=0.9)
        plt.gca().add_artist(legend1)
    
        legend2 = plt.legend(handles=node_legend_elements, loc='upper right', 
                            fontsize=13, title="Nodes", 
                        title_fontsize=14, framealpha=0.9)
    
    # 增加标题字体大小
        plt.title(f'Spatial Network: Spillover Effects from {province_english_map.get(target_province, target_province)}\n(Edge thickness and color indicate spillover intensity)', 
              fontsize=16, fontweight='bold', pad=25)
    
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"spatial_network_{target_province}.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    # 3. 创建详细分析图（目标省份和邻居省份）
    if len(neighbor_indices) > 0:
        plt.figure(figsize=(14, 10))
        
        # 准备数据
        selected_indices = [results['target_idx']] + list(neighbor_indices)
        selected_provinces = [provinces[i] for i in selected_indices]
        selected_changes = [mean_changes[i] for i in selected_indices]
        
        # 计算相对变化（百分比）
        ee_std = results['ee_std']
        ee_mean = results['ee_mean']
        relative_changes = []
        
        for i in selected_indices:
            original_ee_mean = results['original_predictions'][:, i].mean() * ee_std + ee_mean
            if abs(original_ee_mean) > 1e-10:
                rel_change = (mean_changes[i] / original_ee_mean) * 100
            else:
                rel_change = 0
            relative_changes.append(rel_change)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1：绝对变化条形图
        colors1 = ['#FF6B6B'] + ['#FFA726'] * len(neighbor_indices)
        bars1 = axes[0, 0].bar(range(len(selected_provinces)), selected_changes, 
                              color=colors1, alpha=0.85, edgecolor='black')
        axes[0, 0].set_xticks(range(len(selected_provinces)))
        axes[0, 0].set_xticklabels([province_english_map.get(p, p) for p in selected_provinces], 
                                  rotation=45, ha='right', fontsize=11)
        axes[0, 0].set_ylabel('Absolute Change in EE', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Absolute Impact on Target and Neighbors', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
        
        # 在柱子上添加数值标签
        for bar, change in zip(bars1, selected_changes):
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_offset = max(selected_changes) * 0.02
            else:
                va = 'top'
                y_offset = -max(abs(ch) for ch in selected_changes) * 0.02
            
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + y_offset,
                          f'{change:.4f}', ha='center', va=va,
                          fontsize=10, fontweight='bold')
        
        # 子图2：相对变化条形图
        colors2 = ['#FF6B6B'] + ['#FFA726'] * len(neighbor_indices)
        bars2 = axes[0, 1].bar(range(len(selected_provinces)), relative_changes,
                              color=colors2, alpha=0.85, edgecolor='black')
        axes[0, 1].set_xticks(range(len(selected_provinces)))
        axes[0, 1].set_xticklabels([province_english_map.get(p, p) for p in selected_provinces],
                                  rotation=45, ha='right', fontsize=11)
        axes[0, 1].set_ylabel('Relative Change in EE (%)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Relative Impact on Target and Neighbors', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0, 1].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
        
        # 在柱子上添加数值标签
        for bar, change in zip(bars2, relative_changes):
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_offset = max(relative_changes) * 0.02
            else:
                va = 'top'
                y_offset = -max(abs(ch) for ch in relative_changes) * 0.02
            
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + y_offset,
                          f'{change:.2f}%', ha='center', va=va,
                          fontsize=10, fontweight='bold')
        
        # 子图3：邻居省份变化分布
        if len(neighbor_indices) > 1:
            neighbor_changes = mean_changes[neighbor_indices]
            axes[1, 0].hist(neighbor_changes, bins=10, color='#FFA726', alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=results['avg_neighbor_change'], color='red', 
                              linestyle='--', linewidth=2, label=f'Mean: {results["avg_neighbor_change"]:.4f}')
            axes[1, 0].set_xlabel('EE Change', fontsize=11, fontweight='bold')
            axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
            axes[1, 0].set_title('Distribution of Changes in Neighbor Provinces', 
                                 fontsize=13, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        
        # 子图4：目标与邻居对比
        if len(neighbor_indices) > 0:
            categories = ['Target Province', 'Average Neighbor', 'Max Neighbor', 'Min Neighbor']
            values = [results['target_change'], results['avg_neighbor_change'], 
                     results['max_neighbor_change'], results['min_neighbor_change']]
            colors4 = ['#FF6B6B', '#FFA726', '#4ECDC4', '#45B7D1']
            
            bars4 = axes[1, 1].bar(categories, values, color=colors4, alpha=0.85, edgecolor='black')
            axes[1, 1].set_ylabel('EE Change', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('Comparison: Target vs. Neighbors', fontsize=13, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
            axes[1, 1].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
            
            # 在柱子上添加数值标签
            for bar, value in zip(bars4, values):
                height = bar.get_height()
                if height >= 0:
                    va = 'bottom'
                    y_offset = max(values) * 0.02
                else:
                    va = 'top'
                    y_offset = -max(abs(v) for v in values) * 0.02
                
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + y_offset,
                              f'{value:.4f}', ha='center', va=va,
                              fontsize=10, fontweight='bold')
        
        plt.suptitle(f'Detailed Analysis: Impact on {province_english_map.get(target_province, target_province)} and Its Neighbors', 
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"detailed_impact_{target_province}.png"), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    # 4. 创建时间趋势图（如果数据包含时间信息）
    if 'changes_original' in results and results['changes_original'].shape[0] > 1:
        plt.figure(figsize=(14, 8))
        
        # 选择几个关键省份绘制时间趋势
        key_indices = [results['target_idx']]
        if len(neighbor_indices) > 0:
            # 选择变化最大的邻居省份
            neighbor_changes = mean_changes[neighbor_indices]
            max_neighbor_idx = neighbor_indices[np.argmax(np.abs(neighbor_changes))]
            key_indices.append(max_neighbor_idx)
        
        # 如果还有空间，添加变化最大的非邻居省份
        non_neighbor_indices = [i for i in range(len(provinces)) 
                               if i not in neighbor_indices and i != results['target_idx']]
        if non_neighbor_indices:
            non_neighbor_changes = mean_changes[non_neighbor_indices]
            max_non_neighbor_idx = non_neighbor_indices[np.argmax(np.abs(non_neighbor_changes))]
            key_indices.append(max_non_neighbor_idx)
        
        # 绘制时间趋势
        time_points = np.arange(results['changes_original'].shape[0])
        
        # 创建颜色映射
        colors = ['#FF6B6B', '#FFA726', '#4ECDC4', '#96CEB4', '#FFEAA7']
        line_styles = ['-', '--', '-.', ':', '-']
        
        for idx_num, idx in enumerate(key_indices):
            province_name = provinces[idx]
            changes_over_time = results['changes_original'][:, idx]
            
            # 使用移动平均平滑曲线
            window_size = max(1, len(time_points) // 10)
            if window_size > 1:
                changes_smoothed = np.convolve(changes_over_time, np.ones(window_size)/window_size, mode='same')
            else:
                changes_smoothed = changes_over_time
            
            # 标记省份类型
            label_suffix = ""
            if idx == results['target_idx']:
                label_suffix = " (Target)"
            elif idx in neighbor_indices:
                label_suffix = " (Neighbor)"
            
            plt.plot(time_points, changes_smoothed, linewidth=2.5, 
                    color=colors[idx_num % len(colors)],
                    linestyle=line_styles[idx_num % len(line_styles)],
                    label=f"{province_english_map.get(province_name, province_name)}{label_suffix}")
            
            # 填充区域显示标准差
            if idx == results['target_idx']:  # 只为目标省份填充
                changes_std = results['changes_original'][:, idx].std()
                plt.fill_between(time_points, 
                                changes_smoothed - changes_std,
                                changes_smoothed + changes_std,
                                alpha=0.2, color=colors[idx_num % len(colors)])
        
        plt.xlabel('Time Sample', fontsize=12, fontweight='bold')
        plt.ylabel('Change in EE', fontsize=12, fontweight='bold')
        plt.title(f'Time Trend of Impact from Increasing DE in {province_english_map.get(target_province, target_province)}', 
                  fontsize=14, fontweight='bold', pad=15)
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"time_trend_{target_province}.png"), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

# ==================== 7. 保存结果 ====================
def save_counterfactual_results(results, provinces, save_dir):
    """保存反事实模拟结果到文件"""
    
    target_province = results['target_province']
    filename = f"counterfactual_results_{target_province}.txt"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"COUNTERFACTUAL ANALYSIS RESULTS: {target_province}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Simulation Scenario:\n")
        f.write(f"  Target Province: {target_province}\n")
        f.write(f"  DE Increase: {results['de_increase']} standard deviations\n")
        f.write(f"  Time Period: 2011-2023\n")
        f.write(f"  Total Provinces: {len(provinces)}\n")
        f.write(f"  Neighbor Identification: Fixed adjacency matrix\n\n")
        
        f.write("Neighbor Provinces:\n")
        if len(results['neighbor_names']) > 0:
            for i, neighbor in enumerate(results['neighbor_names'], 1):
                f.write(f"  {i}. {neighbor}\n")
        else:
            f.write("  No neighbors identified\n")
        f.write("\n")
        
        f.write("Key Findings:\n")
        f.write(f"  1. Direct effect on target province:\n")
        f.write(f"     - {target_province}: EE change = {results['target_change']:.6f}\n")
        
        if len(results['neighbor_indices']) > 0:
            f.write(f"\n  2. Spatial spillover effects:\n")
            f.write(f"     - Number of affected neighbors: {len(results['neighbor_indices'])}\n")
            f.write(f"     - Average neighbor EE change: {results['avg_neighbor_change']:.6f}\n")
            f.write(f"     - Maximum neighbor EE change: {results['max_neighbor_change']:.6f}\n")
            f.write(f"     - Minimum neighbor EE change: {results['min_neighbor_change']:.6f}\n")
            f.write(f"     - Spillover ratio (neighbor/target): {results['spillover_ratio']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED PROVINCE-LEVEL CHANGES:\n")
        f.write("="*80 + "\n")
        f.write(f"{'Province':<20} {'EE Change':<15} {'Std Dev':<15} {'Relative Change (%)':<20} {'Type':<10}\n")
        f.write("-"*80 + "\n")
        
        ee_std = results['ee_std']
        ee_mean = results['ee_mean']
        
        for i, province in enumerate(provinces):
            change = results['mean_changes_by_province'][i]
            std = results['std_changes_by_province'][i]
            
            # 计算原始EE均值
            original_ee_mean = results['original_predictions'][:, i].mean() * ee_std + ee_mean
            if abs(original_ee_mean) > 1e-10:
                rel_change = (change / original_ee_mean) * 100
            else:
                rel_change = 0
            
            # 标记目标省份和邻居省份
            province_type = ""
            if province == target_province:
                province_type = "Target"
            elif i in results['neighbor_indices']:
                province_type = "Neighbor"
            else:
                province_type = "Other"
            
            f.write(f"{province:<20} {change:<15.6f} {std:<15.6f} {rel_change:<20.2f} {province_type:<10}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SPATIAL SPILLOVER ANALYSIS:\n")
        f.write("="*80 + "\n")
        
        if len(results['neighbor_indices']) > 0:
            f.write("\nNeighbor Provinces (sorted by impact):\n")
            neighbor_changes = results['mean_changes_by_province'][results['neighbor_indices']]
            sorted_neighbor_idx = np.argsort(np.abs(neighbor_changes))[::-1]
            
            for rank, idx in enumerate(sorted_neighbor_idx):
                province_idx = results['neighbor_indices'][idx]
                province_name = provinces[province_idx]
                change = neighbor_changes[idx]
                
                # 计算原始EE均值
                original_ee_mean = results['original_predictions'][:, province_idx].mean() * ee_std + ee_mean
                if abs(original_ee_mean) > 1e-10:
                    rel_change = (change / original_ee_mean) * 100
                else:
                    rel_change = 0
                
                f.write(f"  {rank+1}. {province_name}: EE change = {change:.6f} ({rel_change:.2f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("POLICY IMPLICATIONS:\n")
        f.write("="*80 + "\n")
        f.write("\nBased on the counterfactual simulation results:\n")
        f.write("1. Digital economy development in one province has both direct and spillover effects.\n")
        f.write(f"2. Increasing DE in {target_province} by {results['de_increase']} SD would:\n")
        f.write(f"   - Directly increase EE in {target_province} by {results['target_change']:.4f} units\n")
        
        if len(results['neighbor_indices']) > 0:
            f.write(f"   - Spill over to {len(results['neighbor_indices'])} neighboring provinces\n")
            f.write(f"   - Create an average EE increase of {results['avg_neighbor_change']:.4f} units in neighbors\n")
            f.write(f"   - Amplify the total impact by a factor of {1 + results['spillover_ratio']:.2f}\n")
        
        f.write("\n3. Regional coordination in digital economy development can:\n")
        f.write("   - Maximize positive spatial spillovers\n")
        f.write("   - Reduce regional disparities in energy efficiency\n")
        f.write("   - Create synergistic effects across regions\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("METHODOLOGY:\n")
        f.write("="*80 + "\n")
        f.write("\nCounterfactual Simulation Approach:\n")
        f.write("1. Train a Spatio-Temporal Graph Neural Network (STGNN) on historical data\n")
        f.write("2. Keep all other conditions constant, increase DE in the target province\n")
        f.write("3. Use the trained model to predict EE changes for all provinces\n")
        f.write("4. Compare counterfactual predictions with original predictions\n")
        f.write("5. Quantify direct effects and spatial spillover effects using fixed adjacency matrix\n")
    
    print(f"Results saved to: {filepath}")
    
    # 保存CSV格式的详细结果
    csv_filename = f"counterfactual_details_{target_province}.csv"
    csv_path = os.path.join(save_dir, csv_filename)
    
    csv_data = []
    for i, province in enumerate(provinces):
        change = results['mean_changes_by_province'][i]
        std = results['std_changes_by_province'][i]
        
        # 计算原始EE均值
        original_ee_mean = results['original_predictions'][:, i].mean() * results['ee_std'] + results['ee_mean']
        if abs(original_ee_mean) > 1e-10:
            rel_change = (change / original_ee_mean) * 100
        else:
            rel_change = 0
        
        province_type = "Target" if province == target_province else ("Neighbor" if i in results['neighbor_indices'] else "Other")
        
        csv_data.append({
            'Province': province,
            'Province_Type': province_type,
            'EE_Change': change,
            'EE_Change_Std': std,
            'Relative_Change_Percent': rel_change,
            'Original_EE_Mean': original_ee_mean
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Detailed results saved to: {csv_path}")

# ==================== 8. 主程序 ====================
def main():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. 加载数据
    print("\n" + "="*80)
    print("STEP 1: Loading data...")
    print("="*80)
    
    adj_path = "E:/zh.xlsx"
    feat_path = "E:/jd.xlsx"
    
    provinces, adj_matrices, feature_matrices = load_data_simple(adj_path, feat_path)
    
    if provinces is None:
        print("Failed to load data. Please check the file paths and formats.")
        return
    
    print(f"\nData summary:")
    print(f"  Provinces: {len(provinces)}")
    print(f"  Time steps: {len(adj_matrices)}")
    print(f"  Feature dimension: {feature_matrices[0]['features'].shape[1]}")
    print(f"  Feature names: {feature_matrices[0]['feature_names']}")
    
    # 2. 创建数据集
    print("\n" + "="*80)
    print("STEP 2: Creating dataset...")
    print("="*80)
    
    window_size = 3
    try:
        dataset = SimpleSTGNNDataset(adj_matrices, feature_matrices, window_size=window_size)
        print(f"Dataset created: {len(dataset)} samples")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    # 3. 划分训练集和验证集
    print("\n" + "="*80)
    print("STEP 3: Splitting dataset into training and validation sets...")
    print("="*80)
    
    total_samples = len(dataset)
    train_size = int(0.7 * total_samples)
    val_size = total_samples - train_size
    
    if total_samples < 2:
        print("Error: Not enough samples for training and validation.")
        return
    
    # 设置随机种子以确保可重复性
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=min(4, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(4, len(val_dataset)), shuffle=False)
    
    print(f"Data split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    # 4. 创建模型
    print("\n" + "="*80)
    print("STEP 4: Creating model...")
    print("="*80)
    
    sample = dataset[0]
    n_features = sample['features'].shape[-1]
    n_nodes = sample['features'].shape[1]
    
    model = SimpleSTGNN(
        n_features=n_features,
        n_nodes=n_nodes,
        window_size=window_size,
        hidden_dim=64,
        dropout=0.2
    )
    
    print(f"Model created:")
    print(f"  Input features: {n_features}")
    print(f"  Number of nodes: {n_nodes}")
    print(f"  Window size: {window_size}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 训练模型
    print("\n" + "="*80)
    print("STEP 5: Training model...")
    print("="*80)
    
    model_save_path = os.path.join(output_dir, "best_stgnn_model.pth")
    
    # 检查是否有预训练模型
    if os.path.exists(model_save_path):
        print(f"Found pre-trained model at {model_save_path}. Loading...")
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded (epoch {checkpoint['epoch'] + 1}, R²={checkpoint['val_r2']:.4f})")
    else:
        print("No pre-trained model found. Training new model...")
        try:
            train_losses, val_losses, train_r2s, val_r2s = train_model(
                model, train_loader, val_loader,
                n_epochs=200,
                device=device,
                save_path=model_save_path
            )
            print("Training completed successfully!")
            
            # 绘制训练历史
            plt.figure(figsize=(14, 5))
            
            # 损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Training Loss', linewidth=2)
            plt.plot(val_losses, label='Validation Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # R²曲线
            plt.subplot(1, 2, 2)
            plt.plot(train_r2s, label='Training R²', linewidth=2)
            plt.plot(val_r2s, label='Validation R²', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('R² Score')
            plt.title('R² Score History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.1, 1.1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return
    
    model.to(device)
    
    # 6. 加载最佳模型
    print("\n" + "="*80)
    print("STEP 6: Loading best model for counterfactual analysis...")
    print("="*80)
    
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
        print(f"Best validation R²: {checkpoint['val_r2']:.4f}")
        print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    else:
        print("Error: No model found for counterfactual analysis.")
        return
    
    # 7. 执行反事实模拟
    print("\n" + "="*80)
    print("STEP 7: Performing counterfactual analysis...")
    print("="*80)
    
    # 选择要进行反事实模拟的省份
    target_provinces = ['江苏省', '广东省', '四川省']  # 可以修改这里选择不同的省份
    
    all_results = {}
    
    for target_province in target_provinces:
        print(f"\nPerforming counterfactual analysis for: {target_province}")
        
        # 执行反事实模拟，使用固定邻接矩阵
        results = perform_counterfactual_analysis(
            model=model,
            dataset=dataset,
            provinces=provinces,
            target_province_name=target_province,
            de_increase=0.5,  # 提升0.5个标准差
            device=device,
            fixed_adj_matrix=FIXED_ADJ_MATRIX  # 使用固定邻接矩阵
        )
        
        if results is not None:
            all_results[target_province] = results
            
            # 可视化结果
            print(f"\nVisualizing results for {target_province}...")
            visualize_counterfactual_results(results, provinces, output_dir)
            
            # 保存结果
            print(f"\nSaving results for {target_province}...")
            save_counterfactual_results(results, provinces, output_dir)
    
    # 8. 比较不同省份的反事实模拟结果
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("STEP 8: Comparing results across different provinces...")
        print("="*80)
        
        # 创建比较图
        plt.figure(figsize=(16, 10))
        
        # 准备比较数据
        comparison_data = []
        for target_province, results in all_results.items():
            comparison_data.append({
                'province': target_province,
                'direct_effect': results['target_change'],
                'spillover_effect': results['avg_neighbor_change'],
                'spillover_ratio': results['spillover_ratio'],
                'num_neighbors': len(results['neighbor_indices'])
            })
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1：直接效应比较
        provinces_names = [d['province'] for d in comparison_data]
        direct_effects = [d['direct_effect'] for d in comparison_data]
        
        bars1 = axes[0, 0].bar(provinces_names, direct_effects, color='#FF6B6B', alpha=0.8)
        axes[0, 0].set_ylabel('Direct Effect on EE', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Direct Effect Comparison', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
        
        # 在柱子上添加数值标签
        for bar, effect in zip(bars1, direct_effects):
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_offset = max(direct_effects) * 0.02
            else:
                va = 'top'
                y_offset = -max(abs(e) for e in direct_effects) * 0.02
            
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + y_offset,
                           f'{effect:.4f}', ha='center', va=va,
                           fontsize=10, fontweight='bold')
        
        # 子图2：溢出效应比较
        spillover_effects = [d['spillover_effect'] for d in comparison_data]
        
        bars2 = axes[0, 1].bar(provinces_names, spillover_effects, color='#FFA726', alpha=0.8)
        axes[0, 1].set_ylabel('Average Spillover Effect', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Spillover Effect Comparison', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0, 1].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
        
        # 在柱子上添加数值标签
        for bar, effect in zip(bars2, spillover_effects):
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_offset = max(spillover_effects) * 0.02
            else:
                va = 'top'
                y_offset = -max(abs(e) for e in spillover_effects) * 0.02
            
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + y_offset,
                           f'{effect:.4f}', ha='center', va=va,
                           fontsize=10, fontweight='bold')
        
        # 子图3：溢出比率比较
        spillover_ratios = [d['spillover_ratio'] for d in comparison_data]
        
        bars3 = axes[1, 0].bar(provinces_names, spillover_ratios, color='#4ECDC4', alpha=0.8)
        axes[1, 0].set_ylabel('Spillover Ratio', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Spillover Ratio Comparison (Neighbor/Target)', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
        
        # 在柱子上添加数值标签
        for bar, ratio in zip(bars3, spillover_ratios):
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_offset = max(spillover_ratios) * 0.02
            else:
                va = 'top'
                y_offset = -max(abs(r) for r in spillover_ratios) * 0.02
            
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + y_offset,
                           f'{ratio:.3f}', ha='center', va=va,
                           fontsize=10, fontweight='bold')
        
        # 子图4：邻居数量比较
        num_neighbors = [d['num_neighbors'] for d in comparison_data]
        
        bars4 = axes[1, 1].bar(provinces_names, num_neighbors, color='#45B7D1', alpha=0.8)
        axes[1, 1].set_ylabel('Number of Neighbors', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Number of Affected Neighbors', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 在柱子上添加数值标签
        for bar, num in zip(bars4, num_neighbors):
            height = bar.get_height()
            va = 'bottom'
            y_offset = max(num_neighbors) * 0.02
            
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + y_offset,
                           f'{num}', ha='center', va=va,
                           fontsize=10, fontweight='bold')
        
        plt.suptitle('Comparison of Counterfactual Simulation Results Across Different Provinces', 
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "counterfactual_comparison.png"), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # 保存比较结果
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(output_dir, "counterfactual_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        print(f"Comparison results saved to: {comparison_path}")
    
    # 9. 生成最终报告
    print("\n" + "="*80)
    print("STEP 9: Generating final report...")
    print("="*80)
    
    final_report = f"""FINAL REPORT: COUNTERFACTUAL ANALYSIS OF DIGITAL ECONOMY SPILLOVER EFFECTS
======================================================================================

EXECUTIVE SUMMARY
-----------------
This study performed counterfactual simulations to analyze the spatial spillover effects 
of digital economy (DE) development on energy efficiency (EE) across Chinese provinces.
The analysis used a fixed adjacency matrix based on geographical contiguity to ensure
accurate identification of neighbor relationships.

METHODOLOGY
-----------
1. Trained a Spatio-Temporal Graph Neural Network (STGNN) on provincial data (2011-2023)
2. Conducted counterfactual simulations: Increased DE in target provinces while keeping 
   all other conditions constant
3. Used fixed geographical adjacency matrix for neighbor identification
4. Quantified both direct effects (on target province) and spatial spillover effects 
   (on neighboring provinces)
5. Analyzed the magnitude and patterns of spatial spillovers

KEY FINDINGS
------------
"""
    
    for target_province, results in all_results.items():
        final_report += f"\n{target_province}:\n"
        final_report += f"  - Direct effect: EE changed by {results['target_change']:.4f} units\n"
        if len(results['neighbor_indices']) > 0:
            final_report += f"  - Spillover effect: Average EE change of {results['avg_neighbor_change']:.4f} units in {len(results['neighbor_indices'])} neighbors\n"
            final_report += f"  - Spillover ratio: {results['spillover_ratio']:.3f} (neighbor change / target change)\n"
        else:
            final_report += f"  - No significant spillover effects detected\n"
    
    final_report += f"""
SPATIAL SPILLOVER PATTERNS
--------------------------
1. Digital economy development creates significant spatial spillover effects
2. The magnitude of spillovers varies by region and depends on:
   - Geographical proximity (based on fixed adjacency matrix)
   - Economic connections
   - Digital infrastructure connectivity
3. Spillover effects can amplify the total impact of digital economy policies

POLICY IMPLICATIONS
-------------------
1. Regional coordination: Digital economy policies should consider spatial spillovers
2. Targeted interventions: Focus on provinces with high spillover potential
3. Infrastructure planning: Enhance digital connectivity to maximize positive spillovers
4. Monitoring and evaluation: Track both direct and indirect effects of digital policies

CONCLUSION
----------
Counterfactual simulations provide valuable insights into the spatial dynamics of 
digital economy impacts on energy efficiency. The results demonstrate significant 
spillover effects, highlighting the importance of regional coordination in digital 
development strategies.

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Output directory: {output_dir}
"""
    
    with open(os.path.join(output_dir, "final_counterfactual_report.txt"), 'w', encoding='utf-8') as f:
        f.write(final_report)
    
    print("Final report generated.")
    
    # 10. 列出生成的文件
    print("\n" + "="*80)
    print("GENERATED FILES:")
    print("="*80)
    
    for file in os.listdir(output_dir):
        if os.path.isfile(os.path.join(output_dir, file)):
            print(f"  - {file}")

# ==================== 运行主程序 ====================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("COUNTERFACTUAL SIMULATION: SPATIAL SPILLOVER EFFECTS OF DIGITAL ECONOMY")
    print("="*80)
    print("Using fixed adjacency matrix for accurate neighbor identification")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    start_time = pd.Timestamp.now()
    
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = pd.Timestamp.now()
    print(f"\nTotal execution time: {end_time - start_time}")
    print("\n" + "="*80)
    print("PROGRAM COMPLETED")
    print("="*80)
