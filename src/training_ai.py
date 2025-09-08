#Importação das bibliotecas
import numpy as np
import logging
from extract_and_preprocess_data import PreProcessDataMIT
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#Logging para melhorar a visualização no terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

pre = PreProcessDataMIT()
#Carrega os dados pre-processados da outra classe
x, y, groups = pre.pre_process_data()
le = LabelEncoder()
#Transforma N,S,V e F em 0..3
y_enc = le.fit_transform(y)
#Pega o número total de classes (4)
num_classes = len(np.unique(y_enc))
#Divisão por paciente e dos inputs para treinamento e teste
try:
    gss = GroupShuffleSplit(test_size = 0.2, n_splits = 1, random_state = 42)
    for train_idx, test_idx in gss.split(x, y_enc, groups=groups):
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
    logging.info(f'Divisão Treino:{X_train.shape}, Divisão Teste: {X_test.shape}')
except Exception as e:
    logging.error(f'Erro em realizar a divisão treino/teste')

#Converte os arrays de treino e teste em tensores PyTorch, que é o formato exigido para treino e teste
X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
X_test_t  = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

#Combina os batimentos e as suas respectivas classificações em um único objeto
train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t, y_test_t)
breakpoint()

#Permite processar os batimentos em batch definido no batch_size e melhora a generalização do treino com o shuffle
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

#Definição da classe
class TrainingAICNN:

    def __init__(self):
        pass

if __name__== "__main__":
    train = TrainingAICNN()

