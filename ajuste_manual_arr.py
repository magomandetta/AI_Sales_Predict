import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import datetime

# --- CONFIGURAÇÕES GLOBAIS ---
NOME_ARQUIVO_EXCEL = 'insumos_vendidos_por_dia.xlsx'
INSUMO_ALVO = 'ARR' # O insumo que vamos otimizar
DIAS_VENDA_ALVO = [0, 1, 2, 3, 4, 5, 6] # Diário (todos os dias da semana)

# --- PARÂMETROS DO MODELO LSTM A SEREM AJUSTADOS MANUALMENTE ---
# Altere os valores abaixo para testar diferentes combinações

PARAM_EPOCHS = 100 # Número de épocas de treinamento (mantido fixo como padrão)
PARAM_BATCH_SIZE = 32 # <-- ALtere este valor: [16, 32, 64]
PARAM_UNITS = 64 # Número de neurônios na camada LSTM (mantido fixo como padrão)
PARAM_ACTIVATION = 'relu' # <-- ALtere este valor: ['relu', 'tanh']
PARAM_RECURRENT_DROPOUT = 0.0 # <-- ALtere este valor: [0.0, 0.2, 0.4]
PARAM_BIAS_INITIALIZER = 'zeros' # <-- ALtere este valor: ['zeros', 'ones']
PARAM_LEARNING_RATE = 0.001 # <-- ALtere este valor: [0.001, 0.01]

# --- FUNÇÃO PARA CONSTRUIR O MODELO Keras ---
def build_lstm_model_manual(units=PARAM_UNITS, activation=PARAM_ACTIVATION, 
                            recurrent_dropout=PARAM_RECURRENT_DROPOUT, 
                            bias_initializer=PARAM_BIAS_INITIALIZER, 
                            learning_rate=PARAM_LEARNING_RATE):
    
    model = Sequential()
    model.add(LSTM(units=units, activation=activation, 
                   recurrent_dropout=recurrent_dropout,
                   bias_initializer=bias_initializer, 
                   input_shape=(1, 16), # Assumindo 16 features para ARR
                   return_sequences=False))
    model.add(Dropout(0.2)) 
    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# --- LÓGICA PRINCIPAL ---
def main():
    print(f"--- AJUSTE MANUAL DE HYPERPARÂMETROS para {INSUMO_ALVO} ---")
    print(f"Testando combinação: Activation={PARAM_ACTIVATION}, Rec_Dropout={PARAM_RECURRENT_DROPOUT}, "
          f"Bias_Init={PARAM_BIAS_INITIALIZER}, LR={PARAM_LEARNING_RATE}, Batch_Size={PARAM_BATCH_SIZE}")
    print("Cada execução treinará o modelo 1 vez. Anote os resultados.")
    print("ATENÇÃO: Este script não pode calcular R² e MSE devido a problemas de ambiente.")

    df_full_data = None
    try:
        df_full_data = pd.read_excel(NOME_ARQUIVO_EXCEL)
        df_full_data.columns = df_full_data.columns.str.strip()
        df_full_data['Data'] = pd.to_datetime(df_full_data['Data'], errors='coerce')
        df_full_data.dropna(subset=['Data'], inplace=True)
        for col in df_full_data.columns.drop('Data', errors='ignore'):
            df_full_data[col] = pd.to_numeric(df_full_data[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)

    except Exception as e:
        print(f"ERRO: Não foi possível carregar ou processar o arquivo '{NOME_ARQUIVO_EXCEL}'. {e}")
        return

    # Preparar dados para o insumo alvo
    df_limpo = df_full_data[df_full_data[INSUMO_ALVO] > 0].copy()
    
    if DIAS_VENDA_ALVO is not None and len(DIAS_VENDA_ALVO) < 7:
        df_limpo = df_limpo[df_limpo['Data'].dt.dayofweek.isin(DIAS_VENDA_ALVO)].copy()

    if df_limpo.empty or df_limpo.shape[0] < 20: 
        print(f"ERRO: Poucas amostras para '{INSUMO_ALVO}' ({df_limpo.shape[0]}). Mínimo de 20 para ajuste de LSTM.")
        return

    X_colunas = [col for col in df_limpo.columns if col not in [INSUMO_ALVO, 'Data']]
    if not X_colunas or len(X_colunas) == 0:
        print(f"ERRO: Não há features X para {INSUMO_ALVO}.")
        return

    X_original_df = df_limpo[X_colunas].copy()
    y_original_series = df_limpo[INSUMO_ALVO].copy()
    
    X_original_df = X_original_df.select_dtypes(include=np.number)

    if X_original_df.shape[1] == 0:
        print(f"ERRO: Nenhuma feature numérica válida restou após o processamento.")
        return

    print(f"Número de amostras para {INSUMO_ALVO} após filtragem: {X_original_df.shape[0]}")
    print(f"Número de features para {INSUMO_ALVO}: {X_original_df.shape[1]}")

    # --- DIVISÃO TREINO/TESTE E ESCALONAMENTO RIGOROSO ---
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_original_df, y_original_series, test_size=0.2, random_state=42
    )

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw.values.reshape(-1, 1))

    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw.values.reshape(-1, 1))

    print("Dados divididos e escalonados rigorosamente.")
    
    # --- CONSTRUIR E TREINAR MODELO COM PARÂMETROS ATUAIS ---
    model = build_lstm_model_manual() # Usa os PARAMS definidos no topo do script

    print(f"\nIniciando Treinamento para {INSUMO_ALVO} com os parâmetros atuais...")
    model.fit(X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1]), 
              y_train_scaled, 
              epochs=PARAM_EPOCHS, 
              batch_size=PARAM_BATCH_SIZE, 
              verbose=0) # verbose=0 para menos output

    # --- AVALIAR RESULTADOS (SEM R² E MSE) ---
    print(f"\n--- Treinamento Concluído para {INSUMO_ALVO} ---")
    print("Para avaliar o desempenho, você precisaria rodar o modelo em um ambiente estável com sklearn.metrics.")
    print("O modelo foi treinado com os parâmetros especificados.")
    
if __name__ == '__main__':
    main()