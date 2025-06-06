import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import datetime

# --- CONFIGURAÇÕES GLOBAIS ---
NOME_ARQUIVO_EXCEL = 'backend/insumos_vendidos_por_dia.xlsx'
# Mapeamento dos insumos para seus melhores modelos e dias de venda
# Formato: 'INSUMO': {'tipo_modelo': 'LR' ou 'LSTM', 'dias_venda': [0,1,2,3,4,5,6] ou [dia_num_semana_0a6]}
# O dia da semana vai de 0 (Segunda-feira) a 6 (Domingo)
MODELOS_INSUMOS = {
    'ARR': {'tipo_modelo': 'LSTM', 'dias_venda': [0, 1, 2, 3, 4, 5, 6]}, # Diário
    'FEIJOA': {'tipo_modelo': 'LR', 'dias_venda': [2]}, # >>> CORRIGIDO: QUARTA-FEIRA, MODELO LR
    'BERIN': {'tipo_modelo': 'LSTM', 'dias_venda': [0]}, # Segunda-feira
    'COST': {'tipo_modelo': 'LR', 'dias_venda': [2]}, # Quarta-feira
    'COST S': {'tipo_modelo': 'LR', 'dias_venda': [4]}, # Sexta-feira
    'FRAL': {'tipo_modelo': 'LR', 'dias_venda': [0, 1, 2, 3, 4, 5, 6]}, # Frequente (todos os dias com venda)
    'FRANG': {'tipo_modelo': 'LSTM', 'dias_venda': [0, 1, 2, 3, 4, 5, 6]}, # Diário
    'MAMI': {'tipo_modelo': 'LR', 'dias_venda': [1]}, # Terça-feira
    'MASS': {'tipo_modelo': 'LSTM', 'dias_venda': [0, 1, 2, 3, 4, 5, 6]}, # Diário
    'MOLH': {'tipo_modelo': 'LSTM', 'dias_venda': [0, 1, 2, 3, 4, 5, 6]}, # Quase Diário (todos os dias com venda)
    'MOLH B': {'tipo_modelo': 'LR', 'dias_venda': [3, 4]}, # Quinta e Sexta
    'PEIX': {'tipo_modelo': 'LR', 'dias_venda': [1]}, # Terça-feira
    'POL': {'tipo_modelo': 'LR', 'dias_venda': [3]}, # Quinta-feira
    'TUTU': {'tipo_modelo': 'LR', 'dias_venda': [3]} # Quinta-feira
}

# --- FUNÇÕES AUXILIARES DE MODELAGEM ---

def _prepare_data_for_model(df_full, insumo_alvo, dias_venda_alvo=None):
    """
    Prepara os dados históricos para treinamento do modelo para um dado insumo.
    Realiza filtragem por vendas > 0 e, opcionalmente, por dia da semana.
    Retorna X_original_df_numeric e y_original_series.
    """
    df_temp = df_full.copy()
    
    # Processar coluna alvo
    df_temp[insumo_alvo] = df_temp[insumo_alvo].fillna(0)
    if df_temp[insumo_alvo].dtype == 'object':
        df_temp[insumo_alvo] = df_temp[insumo_alvo].astype(str).str.replace(',', '.', regex=False)
    df_temp[insumo_alvo] = pd.to_numeric(df_temp[insumo_alvo], errors='coerce').fillna(0)

    # Filtrar por vendas > 0 para o insumo alvo
    df_limpo = df_temp[df_temp[insumo_alvo] > 0].copy()

    # Filtrar por dia da semana se especificado
    if dias_venda_alvo is not None and len(dias_venda_alvo) < 7: # Se não for todos os dias da semana
        df_limpo = df_limpo[df_limpo['Data'].dt.dayofweek.isin(dias_venda_alvo)].copy()

    if df_limpo.empty:
        return None, None, "Não há dados históricos suficientes para o insumo ou dia(s) de venda especificado(s)."

    # Separar features (X) e alvo (y)
    X_colunas = [col for col in df_limpo.columns if col not in [insumo_alvo, 'Data']]
    if not X_colunas:
        return None, None, "Não há colunas de features (outros insumos) para este insumo."
    
    X_original_df = df_limpo[X_colunas].copy()
    y_original_series = df_limpo[insumo_alvo].copy()

    # Garantir que features sejam numéricas e preencher NaNs com 0
    for col in X_original_df.columns:
        if X_original_df[col].dtype == 'object':
            X_original_df[col] = X_original_df[col].astype(str).str.replace(',', '.', regex=False)
        X_original_df[col] = pd.to_numeric(X_original_df[col], errors='coerce').fillna(0)
    
    X_numeric_cols_df = X_original_df.select_dtypes(include=np.number)
    
    if X_numeric_cols_df.shape[1] == 0:
        return None, None, "Nenhuma feature numérica válida encontrada após o processamento."

    return X_numeric_cols_df, y_original_series, None


def _train_and_predict_lr(X_train_raw, y_train_raw, X_future_raw):
    """Treina e prevê usando Regressão Linear."""
    
    # Dividir dados brutos (treino) para scaler fit e model fit
    if X_train_raw.shape[0] < 2: # Minimo de 2 amostras para split
        X_tr_raw, y_tr_raw = X_train_raw, y_train_raw
    else:
        X_tr_raw, _, y_tr_raw, _ = train_test_split(
            X_train_raw, y_train_raw, test_size=0.1, random_state=42 # Pequeno split interno
        )

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_tr_raw) # Fit scaler on train data
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_tr_raw.values.reshape(-1, 1))

    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train_scaled.ravel())

    # Predição
    X_future_scaled = scaler_X.transform(X_future_raw.reshape(1, -1)) # Reshape para 2D para scaler
    y_previsao_scaled = model_lr.predict(X_future_scaled)
    estoque_sugerido = scaler_y.inverse_transform(y_previsao_scaled.reshape(-1, 1))[0][0]
    return max(0, estoque_sugerido) # Garantir que o estoque não seja negativo


def _train_and_predict_lstm(X_train_raw, y_train_raw, X_future_raw):
    """Treina e prevê usando LSTM."""
    
    # Dividir dados brutos (treino) para scaler fit e model fit
    if X_train_raw.shape[0] < 2: # Minimo de 2 amostras para split
        X_tr_raw, y_tr_raw = X_train_raw, y_train_raw
    else:
        X_tr_raw, _, y_tr_raw, _ = train_test_split(
            X_train_raw, y_train_raw, test_size=0.1, random_state=42 # Pequeno split interno
        )

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_tr_raw)
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_tr_raw.values.reshape(-1, 1))

    # Reshape X para LSTM (amostras, time_steps=1, features)
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

    # Construir modelo LSTM
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=64, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # Treinar modelo (verbose=0 para não printar muitas linhas)
    batch_size_val = min(32, max(1, X_train_lstm.shape[0] // 4))
    model_lstm.fit(X_train_lstm, y_train_scaled, epochs=100, batch_size=batch_size_val, verbose=0)
    
    # Predição
    X_future_lstm = scaler_X.transform(X_future_raw.reshape(1, -1))
    X_future_lstm = X_future_lstm.reshape((X_future_lstm.shape[0], 1, X_future_lstm.shape[1]))
    
    y_previsao_scaled = model_lstm.predict(X_future_lstm, verbose=0)
    estoque_sugerido = scaler_y.inverse_transform(y_previsao_scaled.reshape(-1, 1))[0][0]
    return max(0, estoque_sugerido)


# --- LÓGICA PRINCIPAL ---

def main():
    print("--- GERADOR DE PREVISÃO DE ESTOQUE MÍNIMO SEMANAL PARA TODOS OS INSUMOS ---")
    print("O modelo utilizará as médias históricas das vendas dos outros insumos para cada dia da semana como base para a previsão futura.")
    print("Atenção: Modelos são retreinados para cada previsão de item/dia, o que pode levar tempo.")

    df_full_data = None
    try:
        df_full_data = pd.read_excel(NOME_ARQUIVO_EXCEL)
        df_full_data.columns = df_full_data.columns.str.strip()
        df_full_data['Data'] = pd.to_datetime(df_full_data['Data'], errors='coerce')
        df_full_data.dropna(subset=['Data'], inplace=True)
        for col in df_full_data.columns.drop('Data', errors='ignore'):
            df_full_data[col] = df_full_data[col].fillna(0)
            if df_full_data[col].dtype == 'object':
                df_full_data[col] = df_full_data[col].astype(str).str.replace(',', '.', regex=False)
            df_full_data[col] = pd.to_numeric(df_full_data[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"ERRO: Não foi possível carregar ou processar o arquivo '{NOME_ARQUIVO_EXCEL}'. {e}")
        return

    # Pedir a data de início da semana ao usuário
    data_str = input("Por favor, digite a data de início da semana que deseja prever (formato予め-MM-DD, ex: 2025-06-02 para uma segunda-feira): ")
    try:
        data_inicio_semana = datetime.datetime.strptime(data_str, "%Y-%m-%d").date()
    except ValueError:
        print("Formato de data inválido. Por favor, use予め-MM-DD.")
        return

    # Preparar a tabela de resultados
    dias_semana_nomes = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
    resultados_previsao = {}

    print(f"\n--- Gerando Previsões para a Semana de {data_inicio_semana} ---")
    print("(Isto pode levar alguns minutos devido ao retreinamento dos modelos...)")

    for i in range(7): # Para cada dia da semana
        data_atual = data_inicio_semana + datetime.timedelta(days=i)
        dia_num_semana = data_atual.weekday() # 0 = Segunda, 6 = Domingo
        nome_dia = dias_semana_nomes[dia_num_semana]
        print(f"\n Calculando previsões para {nome_dia}, {data_atual}...")
        
        previsoes_do_dia_bruto = {} # Armazena resultados para este dia antes de ordenar

        for insumo, config in MODELOS_INSUMOS.items():
            # Verificar se o insumo é vendido neste dia da semana
            if dia_num_semana in config['dias_venda']:
                # Calcular as médias históricas das features para este dia da semana
                df_hist_features_for_day = df_full_data[df_full_data['Data'].dt.dayofweek == dia_num_semana].copy()
                
                all_possible_features_cols = [col for col in df_hist_features_for_day.columns if col not in [insumo, 'Data']]
                
                if df_hist_features_for_day.empty or not all_possible_features_cols:
                    predicted_value = "N/A Hist. Feats" # Sem histórico de features para o dia
                else:
                    X_future_raw_mean = df_hist_features_for_day[all_possible_features_cols].mean().values
                    
                    X_original_df_numeric, y_original_series, error_msg = _prepare_data_for_model(
                        df_full_data, insumo, config['dias_venda']
                    )

                    if X_original_df_numeric is None:
                        predicted_value = "Treino Falho" # Erro na preparação dos dados de treino
                    elif X_original_df_numeric.shape[0] < 10 or X_original_df_numeric.shape[1] == 0:
                        predicted_value = "Dados Insuf." # Poucas amostras para treinar
                    else:
                        try:
                            if config['tipo_modelo'] == 'LR':
                                predicted_value = _train_and_predict_lr(X_original_df_numeric, y_original_series, X_future_raw_mean)
                            elif config['tipo_modelo'] == 'LSTM':
                                predicted_value = _train_and_predict_lstm(X_original_df_numeric, y_original_series, X_future_raw_mean)
                            else:
                                predicted_value = "Modelo Inválido" # Tipo de modelo não mapeado
                        except Exception as e:
                            # print(f"  ERRO ao prever {insumo} para {nome_dia}: {e}")
                            predicted_value = "Erro Previsão"
            else:
                predicted_value = "" # Vazio para itens que não são vendidos naquele dia

            previsoes_do_dia_bruto[insumo] = predicted_value

        resultados_previsao[nome_dia] = previsoes_do_dia_bruto

    # Exibir a tabela de resultados
    df_resultados = pd.DataFrame.from_dict(resultados_previsao, orient='index')
    
    # Reindexar colunas para garantir ordem alfabética dos insumos ou outra ordem desejada
    all_insumos_sorted = sorted(MODELOS_INSUMOS.keys())
    df_resultados = df_resultados.reindex(columns=all_insumos_sorted) 
    
    df_resultados.index.name = 'Dia da Semana'
    df_resultados = df_resultados.reset_index()

    # --- NOVO: Formatação dos valores numéricos com " kg" e "," como separador decimal ---
    for col in df_resultados.columns:
        if col != 'Dia da Semana':
            df_resultados[col] = df_resultados[col].apply(
                lambda x: f"{x:.2f} kg".replace('.', ',') if isinstance(x, (int, float, np.number)) else x
            )

    print("\n\n--- PREVISÃO DE ESTOQUE MÍNIMO SEMANAL (UNIDADES) ---")
    print("Valores vazios indicam que o insumo não é vendido nesse dia da semana.")
    print("Mensagens de erro (e.g., 'Dados Insuf.', 'Erro Previsão') indicam problemas na modelagem para aquele item/dia.")
    print(df_resultados.to_string()) # Usando .to_string() para compatibilidade sem 'tabulate'
    
    print("\nObservação: Para um sistema de produção real, os modelos e scalers seriam treinados uma única vez e salvos/carregados, não retreinados a cada vez.")
    print("Os valores de previsão são a demanda esperada. Um fator de segurança pode ser aplicado (ex: 1.1x) para determinar o estoque mínimo real.")
    print("As previsões são baseadas nas médias históricas dos insumos relacionados para cada dia da semana. Não consideram tendências futuras, feriados, etc.")

if __name__ == '__main__':
    main()