import os
import numpy as np
import pandas as pd
import h5py
import wfdb
from scipy.signal import butter, filtfilt, resample
import logging
from collections import Counter
from sklearn.model_selection import train_test_split
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PTBXLFullPreprocessorCNN2D:
    """
    Preprocessador completo PTB-XL para CNN 2D Híbrida
    - Processa TODOS os 21K registros
    - Mantém sinal COMPLETO de 10s (não segmenta batimentos)
    - Otimizado para diagnóstico de arritmias
    - Preparado para Google Colab GPU
    """
    
    def __init__(self, data_path='./ptbxl/physionet.org/files/ptb-xl/1.0.3/'):
        # Caminhos
        self.data_path = data_path
        self.records_path = os.path.join(data_path, 'records500')
        self.metadata_path = os.path.join(data_path, 'ptbxl_database.csv')
        
        # Configurações do sinal
        self.sampling_rate = 500  # Hz
        self.signal_length = 5000  # 10 segundos @ 500Hz
        
        # Derivações selecionadas (otimizadas para arritmias)
        # Opção A: Todas as 12 derivações (completo)
        self.use_all_leads = True
        self.all_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Opção B: Derivações otimizadas (mais leve)
        self.selected_leads = ['II', 'V1', 'V5']  # Ritmo + Morfologia
        self.lead_indices = {lead: idx for idx, lead in enumerate(self.all_leads)}
        
        # Mapeamento completo de arritmias para AAMI
        self.arrhythmia_to_aami = {
            # NORMAL
            'SR': 'N',      # Ritmo Sinusal Normal
            'SB': 'N',      # Bradicardia Sinusal
            
            # SUPRAVENTRICULAR
            'AFIB': 'S',    # Fibrilação Atrial
            'AFL': 'S',     # Flutter Atrial
            'AT': 'S',      # Taquicardia Atrial
            'AVNRT': 'S',   # Taquicardia por Reentrada Nodal AV
            'AVRT': 'S',    # Taquicardia por Reentrada AV
            'SAAWR': 'S',   # Atrial Rhythm
            'SVTA': 'S',    # Taquicardia Supraventricular
            'ST': 'S',      # Taquicardia Sinusal
            'SA': 'S',      # Arritmia Sinusal
            
            # VENTRICULAR
            'PVC': 'V',     # Contrações Ventriculares Prematuras
            'VPB': 'V',     # Batimentos Ventriculares Prematuros
            'VT': 'V',      # Taquicardia Ventricular
            'VFL': 'V',     # Flutter Ventricular
            'VF': 'V',      # Fibrilação Ventricular
            'BIGU': 'V',    # Bigeminismo Ventricular
            'TRIGU': 'V',   # Trigeminismo Ventricular
            'IVR': 'V'      # Ritmo Idioventricular
        }
        
        # Cache
        self.metadata_df = None
        self.filtered_records = None

    def load_metadata(self):
        """Carrega e filtra metadados para arritmias"""
        logging.info("=== CARREGANDO METADADOS PTB-XL ===")
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadados não encontrados: {self.metadata_path}")
        
        # Carrega CSV
        self.metadata_df = pd.read_csv(self.metadata_path)
        logging.info(f"Total de registros no dataset: {len(self.metadata_df)}")
        
        # Filtra apenas arritmias
        self.filtered_records = self._filter_arrhythmia_records()
        logging.info(f"Registros com arritmias (N,S,V): {len(self.filtered_records)}")
        
        records_normal = self.filtered_records[self.filtered_records['aami_class'] == 'N']
        records_arrit = self.filtered_records[self.filtered_records['aami_class'].isin(['S', 'V'])]

        target_ratio = 2.0
        n_arri = len(records_arrit)
        num = int(n_arri * target_ratio)

        filter_normal_class_rows = records_normal.sample(n=num, random_state=42)
        self.filtered_records = pd.concat([filter_normal_class_rows, records_arrit])
        self.filtered_records = self.filtered_records.sort_values(by='ecg_id', ascending=True)

        logging.info(f'Registros após undersampling: {len(self.filtered_records)}')
        # Estatísticas
        self._print_dataset_statistics()
        return self.filtered_records

    def _filter_arrhythmia_records(self):
        """Filtra registros que contêm arritmias N, S ou V"""
        arrhythmia_records = []
        class_distribution = Counter()
        
        for idx, row in tqdm(self.metadata_df.iterrows(), 
                            total=len(self.metadata_df),
                            desc="Filtrando arritmias"):
            try:
                # Parse dos códigos SCP
                if pd.isna(row['scp_codes']):
                    continue
                    
                scp_codes = ast.literal_eval(row['scp_codes'])
                
                # Verifica se tem arritmia mapeada
                arrhythmia_class = self._get_arrhythmia_class(scp_codes)
                
                if arrhythmia_class:
                    arrhythmia_records.append({
                        'ecg_id': row['ecg_id'],
                        'filename_hr': row['filename_hr'],
                        'scp_code': arrhythmia_class,
                        'aami_class': self.arrhythmia_to_aami[arrhythmia_class],
                        'age': row['age'],
                        'sex': row['sex'],
                        'patient_id': row['patient_id']
                    })
                    class_distribution[self.arrhythmia_to_aami[arrhythmia_class]] += 1
                    
            except Exception as e:
                logging.warning(f"Erro ao processar registro {row['ecg_id']}: {e}")
                continue
        
        logging.info(f"Distribuição de classes AAMI: {dict(class_distribution)}")
        return pd.DataFrame(arrhythmia_records)

    def _get_arrhythmia_class(self, scp_codes):
        """Retorna classe de arritmia prioritária (V > S > N)"""
        # Prioridade: Ventriculares > Supraventriculares > Normal
        
        # 1. Verifica ventriculares (mais graves)
        for code in ['VF', 'VFL', 'VT', 'PVC', 'VPB', 'BIGU', 'TRIGU', 'IVR']:
            if code in scp_codes:
                return code
        
        # 2. Verifica supraventriculares
        for code in ['AFIB', 'AFL', 'AT', 'AVNRT', 'AVRT', 'SAAWR', 'SVTA', 'ST', 'SA']:
            if code in scp_codes:
                return code
        
        # 3. Verifica normais
        for code in ['SR', 'SB']:
            if code in scp_codes:
                return code
        
        return None

    def _print_dataset_statistics(self):
        """Imprime estatísticas detalhadas do dataset"""
        logging.info("\n=== ESTATÍSTICAS DO DATASET ===")
        
        if self.filtered_records is None or len(self.filtered_records) == 0:
            logging.warning("Nenhum registro filtrado disponível")
            return
        
        # Distribuição por classe AAMI
        aami_dist = self.filtered_records['aami_class'].value_counts()
        logging.info("\n📊 Distribuição AAMI:")
        for aami_class, count in aami_dist.items():
            class_name = {'N': 'Normal', 'S': 'Supraventricular', 'V': 'Ventricular'}.get(aami_class, aami_class)
            percentage = (count / len(self.filtered_records)) * 100
            logging.info(f"  {aami_class} ({class_name}): {count} ({percentage:.1f}%)")
        
        # Distribuição por código específico
        code_dist = self.filtered_records['scp_code'].value_counts().head(10)
        logging.info("\n📋 Top 10 códigos específicos:")
        for code, count in code_dist.items():
            percentage = (count / len(self.filtered_records)) * 100
            logging.info(f"  {code}: {count} ({percentage:.1f}%)")
        
        # Demografia
        logging.info(f"\n👥 Pacientes únicos: {self.filtered_records['patient_id'].nunique()}")
        logging.info(f"📅 Idade média: {self.filtered_records['age'].mean():.1f} anos")
        sex_dist = self.filtered_records['sex'].value_counts()
        logging.info(f"⚥ Sexo: {dict(sex_dist)}")

    def bandpass_filter(self, signal, lowcut=0.5, highcut=40, order=4):
        """
        Filtro passa-banda Butterworth
        - Remove ruído de linha de base (<0.5Hz)
        - Remove ruído de alta frequência (>40Hz)
        """
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = butter(order, [low, high], btype='band')
        
        # Aplica filtro em cada derivação
        if len(signal.shape) > 1:
            filtered = np.zeros_like(signal)
            for i in range(signal.shape[1]):
                filtered[:, i] = filtfilt(b, a, signal[:, i])
            return filtered
        else:
            return filtfilt(b, a, signal)

    def load_and_preprocess_single_record(self, record_info):

        try:
            # Monta caminho do arquivo
            filename = record_info['filename_hr']
            file_path = os.path.join(self.data_path, filename)
            # if not os.path.exists(file_path):
            #     return None, None, None
            
            # Carrega sinal (shape: [5000, 12])
            record = wfdb.rdrecord(file_path)
            signal = record.p_signal
            
            # Verifica dimensões
            if signal.shape[0] != self.signal_length:
                # Resample se necessário
                signal = resample(signal, self.signal_length, axis=0)
            
            # Seleciona derivações
            if self.use_all_leads:
                # Usa todas as 12 derivações
                selected_signal = signal  # (5000, 12)
            else:
                # Usa apenas derivações selecionadas
                indices = [self.lead_indices[lead] for lead in self.selected_leads]
                selected_signal = signal[:, indices]
            
            # Aplica filtro passa-banda
            filtered_signal = self.bandpass_filter(selected_signal)
            
            # Normalização por derivação (Z-score)
            normalized_signal = np.zeros_like(filtered_signal)
            for i in range(filtered_signal.shape[1]):
                lead_signal = filtered_signal[:, i]
                mean_val = np.mean(lead_signal)
                std_val = np.std(lead_signal)
                
                if std_val > 0:
                    normalized_signal[:, i] = (lead_signal - mean_val) / std_val
                else:
                    normalized_signal[:, i] = lead_signal - mean_val
            
            # Extrai features temporais globais
            temporal_features = self.extract_global_temporal_features(filtered_signal)
            
            # Label
            label = record_info['aami_class']
            
            return normalized_signal, temporal_features, label
            
        except Exception as e:
            logging.warning(f"Erro ao processar {record_info['ecg_id']}: {e}")
            return None, None, None

    def extract_global_temporal_features(self, signal):
        """
        Extrai features temporais do sinal COMPLETO (não por batimento)
        Importante para diagnóstico de arritmias como AFIB
        """
        features = []
        
        # Usa derivação II para detecção de picos R
        lead_II = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
        
        # Detecção de picos R
        from scipy.signal import find_peaks
        import neurokit2 as nk

        # peaks, _ = find_peaks(lead_II, distance=int(0.6 * self.sampling_rate), 
        #                       height=0.3 * np.max(np.abs(lead_II)))

        signals, info = nk.ecg_peaks(lead_II, sampling_rate=self.sampling_rate)
        peaks = info['ECG_R_Peaks']

        # array([ 575, 1039, 1501, 1966, 2438, 2914, 3395, 3867, 4323, 4805])

        # Features de intervalo RR
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / self.sampling_rate
            
            # Estatísticas RR
            features.extend([
                np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,  # RR médio
                np.std(rr_intervals) if len(rr_intervals) > 0 else 0,   # HRV
                np.max(rr_intervals) - np.min(rr_intervals) if len(rr_intervals) > 0 else 0,  # Range
                len(peaks) / 10  # Frequência cardíaca instantânea (batimentos/10s)
            ])
            
            # Regularidade do ritmo (crucial para AFIB)
            if len(rr_intervals) > 2:
                rr_diff = np.abs(np.diff(rr_intervals))
                rhythm_regularity = np.std(rr_diff) if len(rr_diff) > 0 else 0
                features.append(rhythm_regularity)
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Features espectrais (detecção de ondas F em AFIB)
        fft_vals = np.abs(np.fft.fft(lead_II))
        freq = np.fft.fftfreq(len(lead_II), 1/self.sampling_rate)
        
        # Energia em banda de fibrilação (4-9 Hz)
        fib_band_mask = (freq >= 4) & (freq <= 9)
        # Normalizar pela energia total
        total_energy = np.sum(fft_vals)
        fib_energy = np.sum(fft_vals[fib_band_mask]) / total_energy if total_energy > 0 else 0
        #fib_energy = np.sum(fft_vals[fib_band_mask])
        features.append(fib_energy)
        
        # Features morfológicas globais
        features.extend([
            np.max(signal),
            np.min(signal),
            np.std(signal),
            np.mean(np.abs(signal))
        ])
        
        return np.array(features)

    def preprocess_all_records(self, save_path='./preprocessed_data/', 
                               max_records=None, 
                               validation_split=0.2,
                               test_split=0.1):
        """
        Preprocessa TODOS os registros e salva em formato otimizado
        """
        logging.info("\n=== INICIANDO PRÉ-PROCESSAMENTO COMPLETO ===")
        
        if self.filtered_records is None:
            self.load_metadata()
        # Limita registros se especificado (para testes)
        records_to_process = self.filtered_records
        if max_records:
            records_to_process = self.filtered_records.head(max_records)
            logging.info(f"⚠️  Processando apenas {max_records} registros para teste")
        
        # Listas para armazenar dados
        X_signals = []
        X_temporal = []
        y_labels = []
        patient_ids = []
        
        # Processa cada registro
        successful = 0
        failed = 0
        
        for idx, record in tqdm(records_to_process.iterrows(), 
                               total=len(records_to_process),
                               desc="Processando registros"):
            signal, temporal_features, label = self.load_and_preprocess_single_record(record)
            if signal is not None:
                X_signals.append(signal)
                X_temporal.append(temporal_features)
                y_labels.append(label)
                patient_ids.append(record['patient_id'])
                successful += 1
            else:
                failed += 1
        
        logging.info(f"\n✅ Processados com sucesso: {successful}")
        logging.info(f"❌ Falhas: {failed}")
        
        if successful == 0:
            logging.error("Nenhum registro processado com sucesso!")
            logging.error("Possíveis causas:")
            logging.error("  1. Caminho dos arquivos .dat incorreto")
            logging.error("  2. Arquivos não baixados completamente")
            logging.error("  3. Formato de arquivo incorreto")
            logging.error(f"\nVerificando estrutura de arquivos...")
            logging.error(f"  Data path: {self.data_path}")
            logging.error(f"  Records path: {self.records_path}")
            if len(records_to_process) > 0:
                sample_filename = records_to_process.iloc[0]['filename_hr']
                sample_path = os.path.join(self.data_path, sample_filename)
                logging.error(f"  Exemplo de caminho esperado: {sample_path}")
                logging.error(f"  Arquivo existe? {os.path.exists(sample_path)}")
            return None, None
        
        # Converte para arrays numpy
        X_signals = np.array(X_signals)
        X_temporal = np.array(X_temporal)
        y_labels = np.array(y_labels)
        patient_ids = np.array(patient_ids)
        
        # Prepara formato para CNN 2D
        # Input: (batch, altura, largura, canais) = (batch, derivações, tempo, 1)
        num_leads = X_signals.shape[2]
        X_cnn1d = X_signals
        
        logging.info(f"\n📊 SHAPES FINAIS:")
        logging.info(f"  X_cnn1d: {X_cnn1d.shape} (batch, {num_leads} leads, 5000 samples)")
        logging.info(f"  X_temporal: {X_temporal.shape} (batch, {X_temporal.shape[1]} features)")
        logging.info(f"  y_labels: {y_labels.shape}")
        
        # Codifica labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_labels)
        
        logging.info(f"\n🏷️  Classes codificadas: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        
        # Split estratificado por PACIENTE (evita data leakage)
        from sklearn.model_selection import GroupShuffleSplit
        
        # Split treino/temp
        gss = GroupShuffleSplit(n_splits=1, test_size=(validation_split + test_split), random_state=42)
        train_idx, temp_idx = next(gss.split(X_cnn1d, y_encoded, groups=patient_ids))
        
        # Split validação/teste
        temp_patient_ids = patient_ids[temp_idx]
        temp_val_test_split = test_split / (validation_split + test_split)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=temp_val_test_split, random_state=42)
        val_idx, test_idx = next(gss2.split(X_cnn1d[temp_idx], y_encoded[temp_idx], groups=temp_patient_ids))
        
        # Ajusta índices
        val_idx = temp_idx[val_idx]
        test_idx = temp_idx[test_idx]
        
        # Datasets finais
        datasets = {
            'X_train_cnn': X_cnn1d[train_idx],
            'X_train_temporal': X_temporal[train_idx],
            'y_train': y_encoded[train_idx],
            
            'X_val_cnn': X_cnn1d[val_idx],
            'X_val_temporal': X_temporal[val_idx],
            'y_val': y_encoded[val_idx],
            
            'X_test_cnn': X_cnn1d[test_idx],
            'X_test_temporal': X_temporal[test_idx],
            'y_test': y_encoded[test_idx],
            
            'label_encoder': label_encoder,
            'patient_ids_train': patient_ids[train_idx],
            'patient_ids_val': patient_ids[val_idx],
            'patient_ids_test': patient_ids[test_idx]
        }
        
        # Imprime estatísticas dos splits
        logging.info(f"\n📦 SPLITS:")
        logging.info(f"  Treino: {len(train_idx)} ({len(train_idx)/len(X_cnn1d)*100:.1f}%)")
        logging.info(f"  Validação: {len(val_idx)} ({len(val_idx)/len(X_cnn1d)*100:.1f}%)")
        logging.info(f"  Teste: {len(test_idx)} ({len(test_idx)/len(X_cnn1d)*100:.1f}%)")
        
        # Verifica balanceamento
        for split_name, split_y in [('Treino', y_encoded[train_idx]), 
                                    ('Validação', y_encoded[val_idx]),
                                    ('Teste', y_encoded[test_idx])]:
            dist = Counter(split_y)
            logging.info(f"\n  {split_name} - Distribuição:")
            for class_idx, count in dist.items():
                class_name = label_encoder.inverse_transform([class_idx])[0]
                logging.info(f"    {class_name}: {count} ({count/len(split_y)*100:.1f}%)")
        
        # Salva dados preprocessados
        os.makedirs(save_path, exist_ok=True)
        
        # Salva em formato comprimido
        save_file = os.path.join(save_path, 'ptbxl_cnn1d_preprocessed.npz')
        np.savez_compressed(save_file, **datasets)
        
        logging.info(f"\n💾 Dados salvos em: {save_file}")
        logging.info(f"📦 Tamanho do arquivo: {os.path.getsize(save_file) / (1024**2):.1f} MB")
        
        # Salva metadados adicionais
        import pickle
        metadata = {
            'num_leads': num_leads,
            'lead_names': self.all_leads if self.use_all_leads else self.selected_leads,
            'sampling_rate': self.sampling_rate,
            'signal_length': self.signal_length,
            'num_temporal_features': X_temporal.shape[1],
            'classes': label_encoder.classes_.tolist(),
            'total_records': successful,
            'train_patients': len(np.unique(patient_ids[train_idx])),
            'val_patients': len(np.unique(patient_ids[val_idx])),
            'test_patients': len(np.unique(patient_ids[test_idx]))
        }
        
        with open(os.path.join(save_path, 'preprocessing_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logging.info(f"\n✅ PRÉ-PROCESSAMENTO COMPLETO!")
        return datasets, metadata

    def visualize_sample(self, signal, temporal_features, label):
        """Visualiza um exemplo preprocessado"""
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle(f'ECG Preprocessado - Classe: {label}', fontsize=16)
        
        # Remove dimensão do canal se necessário
        if len(signal.shape) == 2:
            signal = signal.T
        
        lead_names = self.all_leads if self.use_all_leads else self.selected_leads
        
        for i in range(min(12, signal.shape[0])):
            ax = axes[i//3, i%3]
            time = np.arange(signal.shape[1]) / self.sampling_rate
            ax.plot(time, signal[i, :], linewidth=0.8)
            ax.set_title(f'Lead {lead_names[i]}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (normalized)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sample_preprocessed_ecg.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logging.info(f"Temporal features: {temporal_features}")


# ===================================================================
# SCRIPT PRINCIPAL
# ===================================================================
if __name__ == "__main__":
    # Configuração
    DATA_PATH = './ptbxl/physionet.org/files/ptb-xl/1.0.3/'
    SAVE_PATH = './preprocessed_data/'
    
    # Função de debug para verificar estrutura de arquivos
    def debug_file_structure():
        logging.info("=== DEBUG: VERIFICANDO ESTRUTURA DE ARQUIVOS ===")
        
        # Verifica pasta principal
        if not os.path.exists(DATA_PATH):
            logging.error(f"❌ DATA_PATH não existe: {DATA_PATH}")
            logging.info("Tentando caminhos alternativos...")
            
            # Tenta caminhos alternativos comuns
            alternative_paths = [
                './ptbxl/',
                './ptbxl/physionet.org/files/ptb-xl/1.0.3/',
                '/content/ptbxl/',
                '/content/ptbxl/physionet.org/files/ptb-xl/1.0.3/'
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    logging.info(f"✅ Encontrado: {alt_path}")
                    return alt_path
            
            logging.error("Nenhum caminho válido encontrado!")
            return None
        else:
            logging.info(f"✅ DATA_PATH existe: {DATA_PATH}")
        
        # Lista conteúdo
        try:
            contents = os.listdir(DATA_PATH)
            logging.info(f"Conteúdo de {DATA_PATH}:")
            for item in contents[:10]:  # Primeiros 10 itens
                logging.info(f"  - {item}")
        except Exception as e:
            logging.error(f"Erro ao listar {DATA_PATH}: {e}")
        
        # Verifica metadados
        metadata_path = os.path.join(DATA_PATH, 'ptbxl_database.csv')
        if os.path.exists(metadata_path):
            logging.info(f"✅ Metadados encontrados: {metadata_path}")
        else:
            logging.error(f"❌ Metadados NÃO encontrados: {metadata_path}")
        
        # Verifica pasta de registros
        records_path = os.path.join(DATA_PATH, 'records500')
        if os.path.exists(records_path):
            logging.info(f"✅ Pasta records500 existe: {records_path}")
            
            # Conta arquivos .dat
            try:
                subfolders = os.listdir(records_path)
                total_dat = 0
                for subfolder in subfolders[:5]:  # Verifica primeiras 5 subpastas
                    subfolder_path = os.path.join(records_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        dat_files = [f for f in os.listdir(subfolder_path) if f.endswith('.dat')]
                        total_dat += len(dat_files)
                        logging.info(f"  Subpasta {subfolder}: {len(dat_files)} arquivos .dat")
                
                logging.info(f"Total aproximado de arquivos .dat: {total_dat}")
                
                if total_dat == 0:
                    logging.error("❌ NENHUM arquivo .dat encontrado!")
                    logging.error("Você precisa baixar os arquivos de sinal do PTB-XL")
                
            except Exception as e:
                logging.error(f"Erro ao verificar records500: {e}")
        else:
            logging.error(f"❌ Pasta records500 NÃO existe: {records_path}")
        
        return DATA_PATH
    
    # Executa debug
    DATA_PATH = debug_file_structure()
    
    if DATA_PATH is None:
        logging.error("\n❌ Estrutura de arquivos incorreta!")
        logging.error("Por favor, verifique:")
        logging.error("  1. PTB-XL foi baixado completamente")
        logging.error("  2. Caminho está correto")
        logging.error("  3. Arquivos .dat estão em records500/")
        exit(1)
    
    # Para teste rápido, limite registros. Para produção, use None
    MAX_RECORDS = None  # None = processar todos (~21K)
    # MAX_RECORDS = 1000  # Descomente para testar com 1000 registros
    
    # Inicializa preprocessador
    preprocessor = PTBXLFullPreprocessorCNN2D(data_path=DATA_PATH)
    
    # Escolha: Todas as 12 derivações ou apenas 3 otimizadas
    preprocessor.use_all_leads = True  # True = 12 derivações, False = 3 derivações
    
    # Carrega metadados
    preprocessor.load_metadata()
    
    # Preprocessa todos os registros
    datasets, metadata = preprocessor.preprocess_all_records(
        save_path=SAVE_PATH,
        max_records=MAX_RECORDS,
        validation_split=0.15,  # 15% validação
        test_split=0.15         # 15% teste, 70% treino
    )
    
    if datasets is None or metadata is None:
        logging.error("\n❌ PRÉ-PROCESSAMENTO FALHOU!")
        logging.error("Execute o debug acima para identificar o problema.")
        logging.error("\nPara debug manual, tente:")
        logging.error("  1. Verificar se arquivos .dat existem:")
        logging.error(f"     ls {os.path.join(DATA_PATH, 'records500/')}")
        logging.error("  2. Testar com MAX_RECORDS = 10 para isolar o problema")
        exit(1)
        # Visualiza exemplo
        sample_idx = 0
        preprocessor.visualize_sample(
            datasets['X_train_cnn'][sample_idx],
            datasets['X_train_temporal'][sample_idx],
            metadata['classes'][datasets['y_train'][sample_idx]]
        )
    else:
        logging.info("\n🚀 DATASET PRONTO PARA TREINAMENTO!")
        logging.info("Para carregar os dados preprocessados:")
        logging.info(f"  data = np.load('{SAVE_PATH}ptbxl_cnn1d_preprocessed.npz')")
        logging.info(f"  X_train = data['X_train_cnn']")
        logging.info(f"  y_train = data['y_train']")