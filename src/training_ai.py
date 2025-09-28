# requer: torch, sklearn, imblearn, numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from extract_and_preprocess_data import PreProcessDataMIT
import logging
from collections import Counter
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------- 1) Split por paciente otimizado ----------
def group_train_val_test_split(X, y, temporal, groups, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split otimizado para dataset pequeno
    - test_size menor (15% vs 20%) para manter mais dados de treino
    - Estratificação por distribuição de classes por paciente
    """
    unique_groups = np.unique(groups)
    
    # Calcula distribuição de anomalias por paciente para estratificação
    patient_anomaly_ratio = {}
    for group in unique_groups:
        mask = groups == group
        anomaly_count = np.sum(y[mask] == 1)
        total_count = np.sum(mask)
        patient_anomaly_ratio[group] = anomaly_count / total_count
    
    # Ordena pacientes por proporção de anomalias para split mais balanceado
    sorted_patients = sorted(unique_groups, key=lambda x: patient_anomaly_ratio[x])
    
    # Split estratificado
    n_test = max(1, int(len(unique_groups) * test_size))
    n_val = max(1, int(len(unique_groups) * val_size))
    
    # Distribui pacientes alternadamente para manter distribuição similar
    test_groups = sorted_patients[::len(sorted_patients)//n_test][:n_test]
    remaining = [p for p in sorted_patients if p not in test_groups]
    val_groups = remaining[::len(remaining)//n_val][:n_val] if remaining else []
    train_groups = [p for p in sorted_patients if p not in test_groups and p not in val_groups]
    
    # Criar máscaras
    train_mask = np.isin(groups, train_groups)
    val_mask = np.isin(groups, val_groups) if val_groups else np.array([False] * len(groups))
    test_mask = np.isin(groups, test_groups)
    
    X_train, y_train, t_train = X[train_mask], y[train_mask], temporal[train_mask]
    X_val, y_val, t_val = X[val_mask], y[val_mask], temporal[val_mask]
    X_test, y_test, t_test = X[test_mask], y[test_mask], temporal[test_mask]
    
    logging.info(f"Split: Train={len(train_groups)} Val={len(val_groups)} Test={len(test_groups)} pacientes")
    logging.info(f"Distribuição Train: {Counter(y_train)}")
    logging.info(f"Distribuição Val: {Counter(y_val)}")
    logging.info(f"Distribuição Test: {Counter(y_test)}")
    
    return X_train, y_train, t_train, X_val, y_val, t_val, X_test, y_test, t_test

# ---------- 2) Dataset com data augmentation ----------
class ECGDatasetAugmented(Dataset):
    def __init__(self, X, temporal, y, is_training=False, augmentation_prob=0.3):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.temporal = torch.tensor(temporal, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.is_training = is_training
        self.augmentation_prob = augmentation_prob
    
    def __len__(self):
        return len(self.y)
    
    def _augment_signal(self, x):
        """Augmentation conservador para sinais ECG"""
        if not self.is_training or torch.rand(1) > self.augmentation_prob:
            return x
        
        # Noise injection muito suave
        if torch.rand(1) < 0.3:
            noise_level = 0.01 * torch.std(x)
            x = x + torch.randn_like(x) * noise_level
        
        # Amplitude scaling muito conservador
        if torch.rand(1) < 0.2:
            scale = torch.FloatTensor(1).uniform_(0.95, 1.05)
            x = x * scale
        
        return x
    
    def __getitem__(self, idx):
        x = self.X[idx].permute(1, 0)  # (C=1, L)
        t = self.temporal[idx]
        
        # Augmentation apenas no treino
        if self.is_training:
            x = self._augment_signal(x)
        
        return x, t, self.y[idx]

# ---------- 3) Modelo ultra-otimizado anti-overfitting ----------
class CNNWithTemporalOptimized(nn.Module):
    def __init__(self, win_len, n_temporal, n_classes):
        super().__init__()
        
        # CNN muito mais enxuta
        self.cnn = nn.Sequential(
            # Primeira camada: features básicas
            nn.Conv1d(1, 8, kernel_size=15, padding=7),  # Reduzido de 16 para 8
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),  # Pooling mais agressivo
            nn.Dropout(0.4),  # Dropout após primeira camada
            
            # Segunda camada: padrões morfológicos
            nn.Conv1d(8, 16, kernel_size=9, padding=4),  # Reduzido de 32 para 16
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Dropout(0.5)
        )
        self.cnn_out_dim = 16
        
        # MLP temporal ainda mais simples
        self.temporal_mlp = nn.Sequential(
            nn.Linear(n_temporal, 16),  # Reduzido de 32 para 16
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5)  # Dropout aumentado
        )
        
        # Classificador com mais regularização
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out_dim + 16, 32),  # Reduzido de 64 para 32
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),  # BatchNorm adicional
            nn.Dropout(0.7),  # Dropout muito alto
            nn.Linear(32, n_classes)
        )
        
        # Inicialização conservadora
        self._initialize_weights()
        
        # Contador de parâmetros
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Modelo criado: {total_params} parâmetros totais, {trainable_params} treináveis")
    
    def _initialize_weights(self):
        """Inicialização conservadora para evitar overfitting"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # Pesos menores
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, t):
        # CNN para morfologia
        feat = self.cnn(x)
        feat = feat.view(feat.size(0), -1)
        
        # MLP para features temporais
        tfeat = self.temporal_mlp(t)
        
        # Fusão com dropout adicional
        combined = torch.cat([feat, tfeat], dim=1)
        combined = nn.functional.dropout(combined, p=0.3, training=self.training)  # Dropout na fusão
        
        return self.classifier(combined)

# ---------- 4) Loops de treino com monitoramento avançado ----------
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def train_one_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train()
    running_loss, preds, trues = 0.0, [], []
    
    for batch_idx, (x, t, y) in enumerate(dataloader):
        x, t, y = x.to(device), t.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x, t)
        loss = criterion(outputs, y)
        
        # L2 regularization manual adicional para pesos do classificador
        l2_reg = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if 'classifier' in name and 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, 2)
        loss += 1e-4 * l2_reg  # Regularização L2 adicional
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        preds.append(outputs.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
    
    preds = np.vstack(preds).argmax(axis=1)
    trues = np.hstack(trues)
    f1 = f1_score(trues, preds, average='macro', zero_division=0)
    
    return running_loss / len(dataloader.dataset), f1

@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss, preds, trues, probs = 0.0, [], [], []
    
    for x, t, y in dataloader:
        x, t, y = x.to(device), t.to(device), y.to(device)
        outputs = model(x, t)
        loss = criterion(outputs, y)
        
        running_loss += loss.item() * x.size(0)
        
        # Para AUC
        prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        pred = outputs.argmax(dim=1).cpu().numpy()
        
        preds.append(pred)
        trues.append(y.cpu().numpy())
        probs.append(prob)
    
    preds = np.hstack(preds)
    trues = np.hstack(trues)
    probs = np.hstack(probs)
    
    f1 = f1_score(trues, preds, average='macro', zero_division=0)
    
    # Calcula AUC se possível
    try:
        auc = roc_auc_score(trues, probs)
    except:
        auc = 0.0
    
    return running_loss / len(dataloader.dataset), f1, auc, preds, trues, probs

# ---------- 5) Pipeline otimizado ----------
def run_optimized_pipeline():
    logging.info("=== PIPELINE ECG HÍBRIDO ANTI-OVERFITTING ===")
    
    # Pré-processa dados
    preprocess = PreProcessDataMIT()
    X, y, temporal, groups = preprocess.pre_process_data()
    
    # Mapear labels
    class_map = {'Normal': 0, 'Anomaly': 1}
    y_int = np.array([class_map[lab] for lab in y])
    
    logging.info(f"Dataset: {len(X)} amostras, {len(np.unique(groups))} pacientes")
    logging.info(f"Distribuição original: {Counter(y)}")
    
    # Split otimizado por paciente
    X_tr, y_tr, t_tr, X_val, y_val, t_val, X_test, y_test, t_test = group_train_val_test_split(
        X, y_int, temporal, groups
    )
    
    # SMOTE conservador apenas no treino
    logging.info("Aplicando SMOTE conservador...")
    original_count = Counter(y_tr)
    target_ratio = 0.4  # 40% de anomalias vs 60% normais (mais conservador)
    target_anomaly_count = int(original_count[0] * target_ratio / (1 - target_ratio))
    
    # Aplica SMOTE personalizado
    X_tr_res, y_tr_res, t_tr_res = preprocess.apply_smote_global(X_tr, y_tr, t_tr)
    
    # Limita o oversampling se necessário
    anomaly_mask = y_tr_res == 1
    if np.sum(anomaly_mask) > target_anomaly_count:
        anomaly_indices = np.where(anomaly_mask)[0]
        np.random.seed(42)  # Para reprodutibilidade
        keep_indices = np.random.choice(anomaly_indices, target_anomaly_count, replace=False)
        normal_indices = np.where(~anomaly_mask)[0]
        final_indices = np.concatenate([normal_indices, keep_indices])
        
        X_tr_res = X_tr_res[final_indices]
        y_tr_res = y_tr_res[final_indices]
        t_tr_res = t_tr_res[final_indices]
    
    logging.info(f"Após SMOTE otimizado: {Counter(y_tr_res)}")
    
    # Datasets com augmentation
    train_ds = ECGDatasetAugmented(X_tr_res, t_tr_res, y_tr_res, is_training=True)
    val_ds = ECGDatasetAugmented(X_val, t_val, y_val, is_training=False)
    test_ds = ECGDatasetAugmented(X_test, t_test, y_test, is_training=False)
    
    # DataLoaders otimizados
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)  # Batch menor
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # Modelo otimizado
    n_temporal = t_tr_res.shape[1]
    n_classes = len(class_map)
    win_len = X.shape[1]
    model = CNNWithTemporalOptimized(win_len, n_temporal, n_classes).to(device)
    
    # Loss function balanceada mas não extrema
    class_counts = np.unique(y_tr_res, return_counts=True)[1]
    total_samples = class_counts.sum()
    # Pesos menos extremos para evitar overfitting na classe minoritária
    class_weights = np.array([1.0, min(3.0, total_samples / (2 * class_counts[1]))])  # Cap no peso
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    logging.info(f"Class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer com regularização
    optimizer = optim.AdamW(  # AdamW tem melhor regularização
        model.parameters(), 
        lr=5e-4,  # Learning rate menor
        weight_decay=1e-2,  # Weight decay alto
        betas=(0.9, 0.999)
    )
    
    # Schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8)
    
    # Warmup scheduler
    def lr_lambda(epoch):
        if epoch < 5:
            return 0.1 + 0.9 * epoch / 5
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=20, min_delta=0.005)
    
    # Tracking para análise
    train_losses, val_losses, val_f1s, val_aucs = [], [], [], []
    best_val_f1 = 0.0
    
    logging.info("Iniciando treinamento...")
    
    # Loop de treinamento
    for epoch in range(1, 101):  # Mais épocas, mas com early stopping
        # Warmup nos primeiros epochs
        if epoch <= 5:
            warmup_scheduler.step()
        
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_f1, val_auc, _, _, _ = evaluate(model, val_loader, criterion)
        
        # Scheduler principal após warmup
        if epoch > 5:
            scheduler.step(val_f1)
        
        # Tracking
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)
        
        # Logging periódico
        if epoch % 10 == 0 or epoch <= 10:
            logging.info(
                f"Epoch {epoch:2d}: train_loss={train_loss:.4f}, train_f1={train_f1:.4f} | "
                f"val_loss={val_loss:.4f}, val_f1={val_f1:.4f}, val_auc={val_auc:.4f}"
            )
        
        # Salva melhor modelo
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model_optimized.pth")
        
        # Early stopping
        if early_stopping(val_f1, model):
            logging.info(f"Early stopping na época {epoch}. Melhor val_f1: {best_val_f1:.4f}")
            break
    
    # Avaliação final
    logging.info("\n=== AVALIAÇÃO FINAL ===")
    model.load_state_dict(torch.load("best_model_optimized.pth"))
    test_loss, test_f1, test_auc, preds, trues, probs = evaluate(model, test_loader, criterion)
    
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test F1-Score: {test_f1:.4f}")
    logging.info(f"Test AUC: {test_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(trues, preds, target_names=['Normal', 'Anomaly'], zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(trues, preds)
    print(cm)
    
    # Análise de performance
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    logging.info(f"Sensibilidade (Recall Anomalia): {sensitivity:.4f}")
    logging.info(f"Especificidade (Recall Normal): {specificity:.4f}")
    
    # Análise de overfitting
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    overfitting_indicator = final_val_loss - final_train_loss
    
    logging.info(f"\n=== ANÁLISE DE OVERFITTING ===")
    logging.info(f"Train Loss Final: {final_train_loss:.4f}")
    logging.info(f"Val Loss Final: {final_val_loss:.4f}")
    logging.info(f"Diferença (overfitting indicator): {overfitting_indicator:.4f}")
    
    if overfitting_indicator < 0.1:
        logging.info("Baixo risco de overfitting")
    elif overfitting_indicator < 0.3:
        logging.info("Moderado risco de overfitting")
    else:
        logging.info("Alto risco de overfitting - considere mais regularização")
    
    return model, test_f1, test_auc

# ---------- Execução ----------
if __name__ == "__main__":
    model, f1, auc = run_optimized_pipeline()