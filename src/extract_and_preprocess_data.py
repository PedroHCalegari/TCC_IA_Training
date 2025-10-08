# Importação das bibliotecas
import wfdb
import os
import numpy as np
from scipy.signal import butter, filtfilt
import logging
from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

#Logging para melhor visualização no terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
#Definição da classe
class PreProcessDataMIT:

    def __init__(self):
        self.aami_classes = {
            'N': ['N', 'L', 'R', 'e', 'j'],  # Normal
            'S': ['A', 'a', 'J', 'S'],        # Supraventricular
            'V': ['V', 'E'],                  # Ventricular
            'F': ['F'],                       # Fusion
            'Q': ['/', 'f', 'Q']              # Unknown/Other
        }

        self.available_records = [
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
            111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
            122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
            209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
            222, 223, 228, 230, 231, 232, 233, 234
        ]
        self.sampling_rate = 360

    #Função para realizar o download dos arquivos necessários de cada paciente no banco de dados do MIT
    def download_files_mitdb(self):
        #Estrutura de repetição para listar todos os registros disponíveis no MIT
        records = [str(r) for r in self.available_records]
        #Estrutura de repetição para baixar os arquivos com extensão .dat, .hea e .atr de cada paciente e salvar na pasta mitdb criada dentro do for. 
        for record in records:
            try:
                #Cria pasta específica para cada paciente
                dir_patient = f'./mitdb/paciente_{record}'
                #Verifica se o diretório já existe
                if os.path.exists(dir_patient):
                    logging.info(f"Diretório do record {record} já existe, cancelando essa iteração")
                    continue
                #Realiza o download dos arquivos de cada paciente
                wfdb.dl_database('mitdb', dl_dir=dir_patient, records=[record], annotators=['atr'])
                logging.info(f'Registro {record} baixado com sucesso.')
            except Exception as e:
                logging.error(f'Erro ao baixar o registro {record}: {e}')

    #Filtro passa-banda para permitir entre 0,5Hz e 40Hz e remover ruídos
    def bandpass_filter(self, signal, lowcut = 0.5, highcut = 40, order=5):
        try:
            nyq = 0.5 * self.sampling_rate
            b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
            return filtfilt(b, a, signal)
        except Exception as e:
            logging.error(f'Erro em aplicar o filtro passa-banda no sinal desse paciente: {e}')

    #Faz a extração dos dados brutos de um paciente definido como record_id
    def extract_raw_data_each_patient(self, record_id):
        try:
            record_path = f'./mitdb/paciente_{record_id}'
            # Carrega o sinal ECG (.dat + .hea)
            record = wfdb.rdrecord(f'{record_path}/{record_id}')
            signals = record.p_signal
            #Considera apenas o canal 0(MLII)
            signals = signals[:,0]
            # Carrega as anotações (.atr)
            annotation = wfdb.rdann(f'{record_path}/{record_id}', 'atr')
            annotations = annotation.sample
            symbols = annotation.symbol
    
            logging.info(f"Registro do paciente {record_id} carregado com sucesso:")
            logging.info(f"  - Sinal: {signals.shape}")
            logging.info(f"  - Anotações: {len(annotations)} batimentos")
            logging.info(f"  - Duração: {len(signals) / self.sampling_rate} segundos")
            return signals, annotations, symbols
            
        except Exception as e:
            logging.error(f"Erro ao carregar registro {record_id}: {e}")
            return None, None, None
    
    #Segmenta o sinal bruto para considerar a parte mais importante do sinal, as anotações que identificam os picos R e também as normaliza
    def segment_beats_with_temporal_context(self, signal, annotations, symbols, win_before, win_after):
        beats, labels, temporal_features = [], [], []
        
        for i, sample in enumerate(annotations):
            if sample - win_before < 0 or sample + win_after >= len(signal):
                continue
                
            # 1. Morfologia do batimento
            window = signal[sample - win_before : sample + win_after]
            window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)
            
            # 2. CONTEXTO TEMPORAL - Features mais completas
            rr_features = []
            
            # RR intervals
            if i > 0:
                rr_current = (annotations[i] - annotations[i-1]) / self.sampling_rate
                rr_features.append(rr_current)
            else:
                rr_features.append(0.0)
                
            if i > 1:
                rr_previous = (annotations[i-1] - annotations[i-2]) / self.sampling_rate  
                rr_variability = abs(rr_current - rr_previous)
                rr_features.extend([rr_previous, rr_variability])
            else:
                rr_features.extend([0.0, 0.0])
            
            # RR próximo (para contexto completo)
            if i < len(annotations) - 1:
                rr_next = (annotations[i+1] - annotations[i]) / self.sampling_rate
                rr_features.append(rr_next)
            else:
                rr_features.append(0.0)
            
            # Heart rate instantâneo
            if rr_features[0] > 0:
                hr_bpm = 60 / rr_features[0]
                rr_features.append(hr_bpm)
            else:
                rr_features.append(0.0)
            
            # Features morfológicas básicas do batimento
            morpho_features = [
                np.max(window_norm),     # Amplitude máxima
                np.min(window_norm),     # Amplitude mínima  
                np.max(window_norm) - np.min(window_norm),  # Peak-to-peak
                np.std(window_norm),     # Desvio padrão
            ]
            
            # Combinar features temporais + morfológicas
            combined_features = rr_features + morpho_features
                
            beats.append(window_norm)
            labels.append(symbols[i])
            temporal_features.append(combined_features)
            
        return np.array(beats), np.array(labels), np.array(temporal_features)

    #Padronização dos tipos de batimentos conforme classes AAMI
    def map_to_aami(self, labels):
        mapped = []
        try:
            for l in labels:
                found = False
                for key, group in self.aami_classes.items():
                    if l in group:
                        mapped.append(key)
                        found = True
                        break
                if not found:
                    continue
            return np.array(mapped)
        except Exception as e:
            logging.error(f'Erro em fazer o mapeamento para classes AAMI: {e}') 

    def map_to_binary_anomaly(self, labels):
        mapped = []
        for l in labels:
            if l in ['N', 'L', 'R', 'e', 'j']:
                mapped.append('Normal')
            elif l in ['A', 'a', 'J', 'S', 'V', 'E', 'F']:
                mapped.append('Anomaly')
        return np.array(mapped)
    
    #Faz a extração de dados brutos e o pré-processamento do paciente
    def pre_process_patient(self, record_id, win_before=150, win_after=200):
        #Extrai os dados dos arquivos
        signal, annotations, symbols = self.extract_raw_data_each_patient(record_id)
        #Aplica filtro passa-banda
        signal = self.bandpass_filter(signal)
        #Segmenta os batimentos em janelas
        beats, labels, temporal_features = self.segment_beats_with_temporal_context(signal, annotations, symbols, win_before, win_after)
        #Mapeamento para classes AAMI
        mapped_labels = self.map_to_binary_anomaly(labels)
        #Desconsidera classe marcada como desconhecida/outro e filtra apenas os importantes para classificação de arritmia
        valid_idx = [i for i, l in enumerate(mapped_labels) if l != 'Q']
        beats = beats[valid_idx]
        mapped_labels = mapped_labels[valid_idx]
        temporal_features = temporal_features[valid_idx] 
        #Adiciona dimensão (necessária para CNN 1D)
        beats = np.expand_dims(beats, axis=-1)
        groups = np.array([record_id] * len(beats))
        return beats, mapped_labels, temporal_features, groups

    #Estrutura de repetição para realizar o pré-processamento de todos os pacientes e concatenar.
    def pre_process_data(self, apply_smote_global=False):
        X_list, Y_list, T_list, g_list = [], [], [], []
        for record_id in self.available_records:
            try:
                logging.info(f"Iniciando pré-Processamento do paciente {record_id}...")
                X, y, t, g = self.pre_process_patient(record_id)
                if X is not None:
                    X_list.append(X)
                    Y_list.append(y)
                    T_list.append(t)
                    g_list.append(g)
                logging.info(f'Pré-processamento do paciente {record_id} finalizado, iniciando o próximo')
            except Exception as e:
                logging.error(f'Pré-processamento do paciente {record_id} deu erro: {e}')
        X = np.vstack(X_list)
        y = np.hstack(Y_list)
        temporal_features = np.vstack(T_list)
        groups = np.hstack(g_list)

        logging.info('Final do pré-processamento')
        logging.info(f' Shape X (morfologia): {X.shape}')
        logging.info(f' Shape temporal_features: {temporal_features.shape}')
        logging.info(f' Shape Y: {y.shape}')
        logging.info(f' Shape groups: {groups.shape}')
        logging.info(f' Distribuição de classes: {Counter(y)}')
        
        return X, y, temporal_features, groups

    #Reduz a quantidade de classes majoritárias que são marcadas como N (Será verificado a necessidade de utilização com base no treinamento da IA)
    # def undersample(self, X, y):
    #     counts = Counter(y)
    #     min_count = min(counts.values())
    #     X_bal, y_bal = [], []
    #     for cls in counts.keys():
    #         idx = np.where(y == cls)[0]
    #         np.random.shuffle(idx)
    #         idx = idx[:min_count]
    #         X_bal.append(X[idx])
    #         y_bal.append(y[idx])
    #     X_bal = np.vstack(X_bal)
    #     y_bal = np.hstack(y_bal)
    #     return X_bal, y_bal


    #Aumenta as classes minoritárias (diferentes de N) sintetizando batimentos semelhantes (Será verificado a necessidade de utilização com base no treinamento da IA)
    def apply_smote_global(self, X, y, temporal_features):
        try:
            # Remove a dimensão extra para SMOTE
            X_flat = X.reshape((X.shape[0], -1))

            # Combina features morfológicas e temporais
            combined_features = np.hstack([X_flat, temporal_features])

            # Define a nova estratégia de amostragem
            counts = Counter(y)
            minority_class = min(counts, key=counts.get)
            majority_class = max(counts, key=counts.get)

            # Aumenta para 40% da classe majoritária, como no seu exemplo
            target_minority_samples = int(counts[majority_class] * 0.5) 
            
            sampling_strategy = {
                minority_class: target_minority_samples,
                majority_class: counts[majority_class] # Mantém a classe majoritária
            }

            logging.info(f"Aplicando SMOTE com a seguinte estratégia: {sampling_strategy}")

            # Aplica SMOTE no lugar de apenas SMOTE
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(combined_features, y)

            # Separa as features novamente
            n_morpho_features = X_flat.shape[1]
            X_morpho_resampled = X_resampled[:, :n_morpho_features]
            temporal_features_resampled = X_resampled[:, n_morpho_features:]

            # Reconstrói a forma original do X
            X_final = X_morpho_resampled.reshape((X_morpho_resampled.shape[0], X.shape[1], X.shape[2]))

            return X_final, y_resampled, temporal_features_resampled

        except Exception as e:
            logging.error(f'Erro ao aplicar SMOTE global: {e}')
            return X, y, temporal_features



    def plot_signal(self, record_id, n_samples=2000):
        # Extrai os dados brutos
        signal, annotations, symbols = self.extract_raw_data_each_patient(record_id)
        if signal is None:
            logging.error(f"Não foi possível carregar o paciente {record_id}")
            return
        
        # Aplica o filtro passa-banda
        #signal = self.bandpass_filter(signal)

        # Pega os primeiros n_samples
        signal = signal[:n_samples]

        # Converte para tempo em segundos
        time = np.arange(len(signal)) / self.sampling_rate

        # Converte amplitude para mV (assumindo que os dados estão em Volts)
        #signal_mv = signal * 1000  

        # Plot
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal, label="ECG Signal", color="b")
        plt.title(f"ECG Signal - Record {record_id} (First {n_samples} samples)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (V)")
        plt.legend()
        plt.grid(True)
        plt.savefig("ecg_record100.png", dpi=300)
#Define o que vai ser executado ao rodar esse script python
if __name__ == "__main__":
    cd = PreProcessDataMIT()
#     cd.pre_process_data()
    #cd.pre_process_data()
    cd.plot_signal(record_id=100)