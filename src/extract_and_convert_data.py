# Importação das bibliotecas
import wfdb
import os
import numpy as np
from scipy.signal import butter, filtfilt
import logging
from collections import Counter
from imblearn.over_sampling import SMOTE

#Logging para melhor visualização no terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
#Definição da classe
class ConvertDataMIT:

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
    def segment_beats(self, signal, annotations, symbols, win_before, win_after):
        beats, labels = [], []
        #Estrutura de repetição para percorrer todos os picos R do paciente
        try:
            for i, sample in enumerate(annotations):
                #Chegacem para garantir que a janela não ultrapasse o início ou o fim do sinal
                if sample - win_before < 0 or sample + win_after >= len(signal):
                    continue
                #Corte de uma janela centrada no pico R para obter a morfologia completa do batimento (ex: Onda P, QRS e T)
                window = signal[sample - win_before : sample + win_after]
                # Normaliza cada janela (z-score) para a IA focar na morfologia do batimento e não na amplitude absoluta
                window = (window - np.mean(window)) / (np.std(window) + 1e-8)
                beats.append(window)
                labels.append(symbols[i])
            return np.array(beats), np.array(labels)
        except Exception as e:
            logging.error(f'Erro ao fazer a segmentação do sinal deste paciente: {e}')

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
                    mapped.append('Q')  # Se não achar, coloca como "Q" (desconhecido)
            return np.array(mapped)
        except Exception as e:
            logging.error(f'Erro em fazer o mapeamento para classes AAMI: {e}')
    
    #Faz a extração de dados brutos e o pré-processamento do paciente
    def pre_process_patient(self, record_id, win_before=100, win_after=200):
        #Extrai os dados dos arquivos
        signal, annotations, symbols = self.extract_raw_data_each_patient(record_id)
        #Aplica filtro passa-banda
        signal = self.bandpass_filter(signal)
        #Segmenta os batimentos em janelas
        beats, labels = self.segment_beats(signal, annotations, symbols, win_before, win_after)
        #Mapeamento para classes AAMI
        labels = self.map_to_aami(labels)
        #Adiciona dimensão (necessária para CNN 1D)
        beats = np.expand_dims(beats, axis=-1)
        groups = np.array([record_id] * len(beats))
        return beats, labels, groups

    #Estrutura de repetição para realizar o pré-processamento de todos os pacientes e concatenar.
    def pre_process_data(self):
        X_list, Y_list, g_list = [], [], []
        for record_id in self.available_records:
            try:
                logging.info(f"Iniciando pré-Processamento do paciente {record_id}...")
                X, y, g = self.pre_process_patient(record_id)
                if X is not None:
                    X_list.append(X)
                    Y_list.append(y)
                    g_list.append(g)
                logging.info(f'Pré-processamento do paciente {record_id} finalizado, iniciando o próximo')
            except Exception as e:
                logging.error(f'Pré-processamento do paciente {record_id} deu erro: {e}')
        X = np.vstack(X_list)
        y = np.hstack(Y_list)
        groups = np.hstack(g_list)

        loggin.info('Final do pré-processamento')
        loggin.info(f' Shape X: {X.shape}')
        loggin.info(f' Shape Y: {y.shape}')
        loggin.info(f' Shape groups: {groups.shape}')
        return X, y, groups

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
    #def oversample(self, X, y):
        # # Flatten para 2D
        # X_flat = X.reshape((X.shape[0], -1))
        # sm = SMOTE(random_state=42)
        # X_res, y_res = sm.fit_resample(X_flat, y)
        # # Volta para 3D
        # X_res = X_res.reshape((X_res.shape[0], X.shape[1], X.shape[2]))
        # return X_res, y_res

#Define o que vai ser executado
if __name__ == "__main__":
    cd = ConvertDataMIT()
    #cd.download_files_mitdb()
    cd.pre_process_data()
