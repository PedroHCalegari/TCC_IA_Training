# Importação das bibliotecas
import wfdb
import os
import numpy as np

#Definição da classe
class convert_data:

    #Até o momento não será necessário
    def __init__(self):
        pass

    #Função para realizar o download dos arquivos necessários de cada paciente no banco de dados do MIT
    #Por questão de otimização de tempo e processamento, só é necessário rodar uma única vez essa função (Já que o database não é atualizado).

    def download_files_mitdb(self):
        #Estrutura de repetição para listar todos os possíveis registros disponíveis no MIT
        records = [str(r) for r in range(100, 235)] 
        #Estrutura de repetição para baixar os arquivos com extensão .dat, .hea e .atr de cada paciente e salvar na pasta mitdb criada dentro do for. 
        for record in records:
            try:
                wfdb.dl_database('mitdb', dl_dir=f'./mitdb/paciente_{record}', records=[record], annotators=['atr'])
                print(f'Registro {record} baixado com sucesso.')
            except Exception as e:
                print(f'Erro ao baixar o registro {record}: {e}')

    

#Define o que vai ser executado
if __name__ == "__main__":
    cd = convert_data()
    #cd.download_files_mitdb()