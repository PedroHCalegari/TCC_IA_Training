# Importação das bibliotecas
import wfdb
import os
import numpy as np

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
                    print(f"Diretório do record {record} já existe, cancelando essa iteração")
                    continue
                #Realiza o download dos arquivos de cada paciente
                wfdb.dl_database('mitdb', dl_dir=dir_patient, records=[record], annotators=['atr'])
                print(f'Registro {record} baixado com sucesso.')
            except Exception as e:
                print(f'Erro ao baixar o registro {record}: {e}')

    #Filtro passa-banda para permitir entre 0,5Hz e 40Hz e remover ruídos
    def bandpass_filter(self, signal, lowcut = 0.5, highcut = 40, order=5):
        nyq = 0.5 * self.sampling_rate
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        return filtfilt(b, a, signal)

    #Faz a extração dos dados brutos de um paciente definido como record_id
    def extract_raw_data_each_patient(self, record_id):
        try:
            record_path = f'./mitdb/paciente_{record_id}'
            # Carrega o sinal ECG (.dat + .hea)
            record = wfdb.rdrecord(f'{record_path}/{record_id}')
            signals = record.p_signal
            
            # Carrega as anotações (.atr)
            annotation = wfdb.rdann(f'{record_path}/{record_id}', 'atr')
            annotations = annotation.sample
            symbols = annotation.symbol
    
            print(f"Registro {record_id} carregado com sucesso:")
            print(f"  - Sinal: {signals.shape}")
            print(f"  - Anotações: {len(annotations)} batimentos")
            print(f"  - Duração: {len(signals) / self.sampling_rate} segundos")
            return signals, annotations, symbols
            
        except Exception as e:
            print(f"Erro ao carregar registro {record_id}: {e}")
            return None, None, None

    #Faz uma estrutura de repetição para realizar a extração dos dados brutos e o pré-processamento de todos os pacientes
    def pre_process_data(self):
        for record_id in self.available_records:
            signals, annotations, symbols = self.extract_raw_data_each_patient(record_id)


    
#Define o que vai ser executado
if __name__ == "__main__":
    cd = ConvertDataMIT()
    cd.download_files_mitdb()
    cd.pre_process_data()