# AI_Sales_Predict

**Previsão de Vendas de Itens de Restaurante com Redes Neurais LSTM**

Este projeto utiliza redes neurais LSTM (Long Short-Term Memory) para prever as vendas de itens de restaurante com base em dados históricos. O objetivo é auxiliar na otimização de inventário e melhorar a tomada de decisões.

## 📊 Objetivo

Desenvolver um modelo de previsão de vendas utilizando redes neurais LSTM para:

- Estimar a demanda futura de itens de restaurante.
- Auxiliar na gestão de estoque e planejamento de compras.
- Melhorar a eficiência operacional e a satisfação do cliente.

## 🧠 Tecnologias Utilizadas

- **Python**: Linguagem principal para desenvolvimento.
- **Pandas**: Manipulação e análise de dados.
- **NumPy**: Cálculos numéricos.
- **Matplotlib / Seaborn**: Visualização de dados.
- **TensorFlow / Keras**: Implementação do modelo LSTM.
- **Scikit-learn**: Pré-processamento e avaliação de modelos.

## 🚀 Como Executar

1. Clone o repositório:

   ```bash
   git clone https://github.com/magomandetta/AI_Sales_Predict.git
   cd AI_Sales_Predict
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Abra o Jupyter Notebook desejado:

   ```bash
   jupyter notebook
   ```

4. Execute as células do notebook para reproduzir o processo de previsão de vendas.

## 📈 Descrição dos Notebooks

* **AI\_Predict.ipynb**: Implementação principal do modelo LSTM para previsão de vendas.
* **Consolida\_Itens.ipynb**: Consolidação e preparação dos dados de itens.
* **LSTM\_LinearRegression.ipynb**: Comparação entre LSTM e regressão linear para previsão de vendas.
* **PreFixos\_Vendas.ipynb**: Análise de prefixos de vendas e seu impacto nas previsões.

## 📊 Conjunto de Dados

O projeto utiliza os seguintes arquivos de dados:

* **ItensPeriodicos.xlsx**: Dados periódicos dos itens.
* **ItensVendidosDia.xlsx**: Vendas diárias dos itens.
* **Itens\_Agrupados\_com\_Dia\_Semana.xlsx**: Vendas agrupadas por dia da semana.
* **Venda\_Itens\_Data\_certo.xlsx**: Dados de vendas corrigidos.

## 📈 Métricas de Avaliação

O desempenho do modelo é avaliado utilizando as seguintes métricas:

* **Erro Quadrático Médio (MSE)**: Mede a média dos quadrados dos erros.
* **Coeficiente de Determinação (R²)**: Indica a proporção da variabilidade dos dados explicada pelo modelo.
