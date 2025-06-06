from flask import Flask, request, jsonify
from flask_cors import CORS 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import os

# --- CONFIGURAÇÕES E MAPEAMENTO DE MODELOS (DO SEU SCRIPT DE PREVISÃO) ---
NOME_ARQUIVO_EXCEL = 'insumos_vendidos_por_dia.xlsx' # O Excel deve estar na mesma pasta do app.py (backend/)
# Mapeamento dos insumos para seus melhores modelos e dias de venda
# O dia da semana vai de 0 (Segunda-feira) a 6 (Domingo)
MODELOS_INSUMOS = {
    'ARR': {'tipo_modelo': 'LSTM', 'dias_venda': [0, 1, 2, 3, 4, 5, 6]}, # Diário
    'FEIJOA': {'tipo_modelo': 'LR', 'dias_venda': [2]}, # >>> CORRIGIDO: QUARTA-FEIRA, MODELO LR
    'BERIN': {'tipo_modelo': 'LSTM', 'dias_venda': [0]}, # Segunda-feira
    'COST': {'tipo_modelo': 'LR', 'dias_venda': [2]}, # Quarta-feira
    'COST S': {'tipo_modelo': 'LR', 'dias_venda': [4]}, # Sexta-feira
    'FRAL': {'tipo_modelo': 'LR', 'dias_venda': [0, 1, 2, 3, 4, 5, 6]}, # Frequente (consideramos todos os dias que tiver vendas)
    'FRANG': {'tipo_modelo': 'LSTM', 'dias_venda': [0, 1, 2, 3, 4, 5, 6]}, # Diário
    'MAMI': {'tipo_modelo': 'LR', 'dias_venda': [1]}, # Terça-feira
    'MASS': {'tipo_modelo': 'LSTM', 'dias_venda': [0, 1, 2, 3, 4, 5, 6]}, # Diário
    'MOLH': {'tipo_modelo': 'LSTM', 'dias_venda': [0, 1, 2, 3, 4, 5, 6]}, # Quase Diário (consideramos todos os dias que tiver vendas)
    'MOLH B': {'tipo_modelo': 'LR', 'dias_venda': [3, 4]}, # Quinta e Sexta
    'PEIX': {'tipo_modelo': 'LR', 'dias_venda': [1]}, # Terça-feira
    'POL': {'tipo_modelo': 'LR', 'dias_venda': [3]}, # Quinta-feira
    'TUTU': {'tipo_modelo': 'LR', 'dias_venda': [3]} # Quinta-feira
}

# --- FUNÇÕES DE MODELAGEM (COPIADAS DO SEU SCRIPT DE PREVISÃO) ---
# >>>>> IMPORTANTE: VOCÊ DEVE COLAR AQUI ABAIXO AS DEFINIÇÕES COMPLETAS DAS FUNÇÕES:
# >>>>> - _prepare_data_for_model
# >>>>> - _train_and_predict_lr
# >>>>> - _train_and_predict_lstm
# >>>>> - build_lstm_model_api (se você usou um nome diferente como build_lstm_model_manual, renomeie aqui ou ajuste a chamada)
# >>>>>   Elas estão no seu script 'previsao_estoque_semanal.py'.

# Exemplo (NÃO COPIE ESTES COMENTÁRIOS, APENAS O CÓDIGO DA FUNÇÃO ABAIXO):
# def _prepare_data_for_model(df_full, insumo_alvo, dias_venda_alvo=None):
#     # ... Cole o código completo da função aqui ...
#     pass # Remova esta linha 'pass'

# def _train_and_predict_lr(X_train_raw, y_train_raw, X_future_raw):
#     # ... Cole o código completo da função aqui ...
#     pass # Remova esta linha 'pass'

# def _train_and_predict_lstm(X_train_raw, y_train_raw, X_future_raw):
#     # ... Cole o código completo da função aqui ...
#     pass # Remova esta linha 'pass'

# def build_lstm_model_api(units=64, activation='relu', recurrent_dropout=0.0, bias_initializer='zeros', learning_rate=0.001):
#     # ... Cole o código completo da função aqui ...
#     pass # Remova esta linha 'pass'

# --- FIM DA SEÇÃO DAS FUNÇÕES DE MODELAGEM ---


# --- APLICATIVO FLASK ---
app = Flask(__name__)
CORS(app) # Habilita CORS para permitir que o frontend React acesse a API

# Carregar o DataFrame completo uma única vez quando a API inicia
PATH_TO_DATA = NOME_ARQUIVO_EXCEL # Assumindo que o Excel está na mesma pasta que app.py (backend/)

df_full_data_global = None
try:
    df_full_data_global = pd.read_excel(PATH_TO_DATA)
    df_full_data_global.columns = df_full_data_global.columns.str.strip()
    df_full_data_global['Data'] = pd.to_datetime(df_full_data_global['Data'], errors='coerce')
    df_full_data_global.dropna(subset=['Data'], inplace=True)
    for col in df_full_data_global.columns.drop('Data', errors='ignore'):
        df_full_data_global[col] = pd.to_numeric(df_full_data_global[col].astype(str).replace(',', '.', regex=False), errors='coerce').fillna(0)
    print("DataFrame carregado e pré-processado na inicialização da API.")
except Exception as e:
    print(f"ERRO CRÍTICO ao carregar o DataFrame global na inicialização da API: {e}")
    df_full_data_global = None # Garante que df_full_data_global seja None em caso de erro

@app.route('/')
def home():
    return "API de Previsão de Estoque funcionando!"

@app.route('/prever_estoque', methods=['POST'])
def prever_estoque():
    if df_full_data_global is None:
        return jsonify({"error": "Dados históricos não puderam ser carregados pela API."}), 500

    data_inicio_str = request.json.get('data_inicio_semana') # Espera JSON com 'data_inicio_semana'
    if not data_inicio_str:
        return jsonify({"error": "Data de início da semana não fornecida."}), 400

    try:
        data_inicio_semana = datetime.datetime.strptime(data_inicio_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Formato de data inválido. Use AAAA-MM-DD."}), 400

    dias_semana_nomes = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
    resultados_previsao = {}

    print(f"\n--- API: Gerando Previsões para a Semana de {data_inicio_semana} ---")

    for i in range(7):
        data_atual = data_inicio_semana + datetime.timedelta(days=i)
        dia_num_semana = data_atual.weekday()
        nome_dia = dias_semana_nomes[dia_num_semana]
        
        previsoes_do_dia_bruto = {}

        for insumo, config in MODELOS_INSUMOS.items():
            if dia_num_semana in config['dias_venda']:
                df_hist_features_for_day = df_full_data_global[df_full_data_global['Data'].dt.dayofweek == dia_num_semana].copy()
                all_possible_features_cols = [col for col in df_hist_features_for_day.columns if col not in [insumo, 'Data']]
                
                if df_hist_features_for_day.empty or not all_possible_features_cols:
                    predicted_value = "N/A Hist. Feats"
                else:
                    X_future_raw_mean = df_hist_features_for_day[all_possible_features_cols].mean().values
                    
                    X_original_df_numeric, y_original_series, error_msg = _prepare_data_for_model(
                        df_full_data_global, insumo, config['dias_venda']
                    )

                    if X_original_df_numeric is None or X_original_df_numeric.shape[0] < 10 or X_original_df_numeric.shape[1] == 0:
                        predicted_value = "Dados Insuf."
                    else:
                        try:
                            if config['tipo_modelo'] == 'LR':
                                predicted_value = _train_and_predict_lr(X_original_df_numeric, y_original_series, X_future_raw_mean)
                            elif config['tipo_modelo'] == 'LSTM':
                                # Note: Otimização de parâmetros do LSTM (build_lstm_model_api)
                                # é um ponto a ser refinado para um sistema de produção.
                                predicted_value = _train_and_predict_lstm(X_original_df_numeric, y_original_series, X_future_raw_mean)
                            else:
                                predicted_value = "Modelo Inválido"
                        except Exception as e:
                            print(f"  ERRO de previsão para {insumo} em {nome_dia}: {e}")
                            predicted_value = "Erro Previsão"
            else:
                predicted_value = "" 

            previsoes_do_dia_bruto[insumo] = predicted_value

        resultados_previsao[nome_dia] = previsoes_do_dia_bruto

    df_resultados = pd.DataFrame.from_dict(resultados_previsao, orient='index')
    all_insumos_sorted = sorted(MODELOS_INSUMOS.keys())
    df_resultados = df_resultados.reindex(columns=all_insumos_sorted) 
    df_resultados.index.name = 'Dia da Semana'
    df_resultados = df_resultados.reset_index()

    for col in df_resultados.columns:
        if col != 'Dia da Semana':
            df_resultados[col] = df_resultados[col].apply(
                lambda x: f"{x:.2f} kg".replace('.', ',') if isinstance(x, (int, float, np.number)) else x
            )
    
    return jsonify(df_resultados.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)