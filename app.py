
import streamlit as st
from PIL import Image
import pandas as pd

#yfin.pdr_override()
import numpy as np
from matplotlib import pyplot as plt
import datetime
import warnings
warnings.filterwarnings("ignore")
from datetime import date

#import prophet
from prophet import Prophet

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np
#np.float_ = np.float64

#import pmdarima as pm

import openpyxl
file = 'Consumo x Previsão.xlsx'
df = pd.read_excel(file)

# Filtra as colunas de data
date_columns = [col for col in df.columns if isinstance(col, datetime.datetime)]

# Filtra as colunas de data
date_columns = [col for col in df.columns if isinstance(col, datetime.datetime)]

# Converte o DataFrame para o formato longo (long format) para reorganizar as colunas de data
df_melted = df.melt(id_vars=[col for col in df.columns if col not in date_columns],
                    value_vars=date_columns,
                    var_name='mes-ano', value_name='valor')

# Convert 'mes-ano' column to datetime objects
df_melted['mes-ano'] = pd.to_datetime(df_melted['mes-ano'])

# Now you can use .dt.strftime
df_melted['mes-ano'] = df_melted['mes-ano'].dt.strftime('%Y-%m')

df_melted.dropna(inplace=True)

descricoes = df_melted['Descrição'].to_list()
codigos    = df_melted['Código'].to_list()
#st.write('Criar dicionario descricao:codigo')
# Cria o dicionário
descricao_para_codigo = dict(zip(descricoes, codigos))
#st.write(descricao_para_codigo)
# Exemplo de uso
#print(descricao_para_codigo["Produto A"])  # Saída: 101

def forecast_xgboost(model, df, steps=12):
    """
    Realiza uma previsão recursiva de múltiplos passos à frente com XGBoost.
    
    Parâmetros:
    - model: modelo treinado XGBoost
    - df: dataframe com os dados de entrada
    - steps: número de meses a prever
    
    Retorna:
    - Lista de previsões para os próximos 'steps' meses
    """
    
    #st.table(df)
    
    # Lista para armazenar as previsões
    predictions = []
    
    # Obtém o último valor disponível para cada coluna lag
    last_mes = df['mes'].iloc[-1]
    last_ano = df['ano'].iloc[-1]
    last_valor_lag1 = df['valor'].iloc[-1]
    last_valor_lag2 = df['valor'].iloc[-2]


    #st.write('teste1')
    for i in range(1, steps + 1):
        # Atualiza o próximo mês e ano
        if last_mes == 12:
            next_mes = 1
            next_ano = last_ano + 1
        else:
            next_mes = last_mes + 1
            next_ano = last_ano

        # Cria o input para o próximo passo
        input_data = pd.DataFrame({
            'mes': [next_mes],
            'ano': [next_ano],
            'valor_lag1': [last_valor_lag1],
            'valor_lag2': [last_valor_lag2]
        })
        #st.write('input_data')
        #st.write(input_data)
        # Faz a previsão
        next_pred = model.predict(input_data)[0]
        
        # Armazena a previsão
        predictions.append(next_pred)
        #st.write('teste2')
        
        # Atualiza os valores lag para o próximo passo
        last_valor_lag2 = last_valor_lag1
        last_valor_lag1 = next_pred
        last_mes = next_mes
        last_ano = next_ano
        
        #st.write('predictions')
        #st.write(predictions)
    return predictions

def training_xgboost(df):
    # Prepara o dataset
    df['mes'] = pd.to_datetime(df['mes-ano']).dt.month
    df['ano'] = pd.to_datetime(df['mes-ano']).dt.year

    # Criando lag features para capturar valores passados
    df['valor_lag1'] = df['valor'].shift(1)
    df['valor_lag2'] = df['valor'].shift(2)

    # Remove valores nulos criados pelas lag features
    df.dropna(inplace=True)

    # Separando features e target
    X = df[['mes', 'ano', 'valor_lag1', 'valor_lag2']]
    y = df['valor']

    # Configura o modelo e a divisão de série temporal
    model = XGBRegressor()
    tscv = TimeSeriesSplit(n_splits=5)

    # Avalia o modelo com Cross-Validation
    mse_scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    #print("MSE médio:", np.mean(mse_scores))

    # Faz previsões para os próximos meses (exemplo de 12 passos)
    #future_X = X.tail(12).copy()  # Adapte para novos meses, se necessário
    #predictions = model.predict(future_X)
    #print(predictions)
    #st.table(df)
    #st.write(model)
    return df, model

def plot_xgboost(df, predictions):
    #st.subheader('plot_xgboost')
    #st.table(df)
    
    df['mes-ano'] = pd.to_datetime(df['mes-ano'], format='%Y-%m', errors='coerce')
    # Extraindo previsões para 3, 6, 9 e 12 meses
    #model = training_xgboost(df)
    # Exemplo de uso para prever os próximos 12 meses
    predicoes_12_meses = predictions
    
    previsao_1_meses = predicoes_12_meses[0]
    previsao_3_meses = predicoes_12_meses[2]   # índice 2 para o terceiro mês
    previsao_6_meses = predicoes_12_meses[5]   # índice 5 para o sexto mês
    previsao_9_meses = predicoes_12_meses[8]   # índice 8 para o nono mês
    previsao_12_meses = predicoes_12_meses[11] # índice 11 para o décimo segundo mês

    # Obtendo a última data do DataFrame
    ultima_data = df['mes-ano'].iloc[-1]

    print('Ultima data:',ultima_data)
    # Criando um DataFrame para as previsões
    previsao_data = {
    'mes-ano': [
        ultima_data + pd.DateOffset(months=1),
        ultima_data + pd.DateOffset(months=3),
        ultima_data + pd.DateOffset(months=6),
        ultima_data + pd.DateOffset(months=9),
        ultima_data + pd.DateOffset(months=12)
    ],
    'valor': [
        previsao_1_meses,
        previsao_3_meses,  # valor previsto para 3 meses
        previsao_6_meses,  # valor previsto para 6 meses
        previsao_9_meses,  # valor previsto para 9 meses
        previsao_12_meses  # valor previsto para 12 meses
    ]
}

    # Convertendo para DataFrame
    df_previsao = pd.DataFrame(previsao_data)
    #st.table(df_previsao)

    # Adicionando as previsões ao DataFrame original
    #df = pd.concat([df, df_previsao], ignore_index=True)

    # Drop rows with invalid dates (NaT)
    df = df.dropna(subset=['mes-ano'])
    #st.table(df)

    # Exibindo o DataFrame atualizado
    #st.table(df)
    #st.table(df_previsao)

    # Plotando os valores históricos e as previsões
    plt.figure(figsize=(12, 6))
    plt.plot(df['mes-ano'], df['valor'], label='Valores Atuais', color='blue', marker='o')
    plt.axvline(x=ultima_data, color='green', linestyle='--', label='Início das Previsões')
    plt.plot(df_previsao['mes-ano'].iloc[-5:], df_previsao['valor'].iloc[-5:], color='red', label='Previsões', marker='o')
    plt.title('Valores Atuais e Previsões para Próximos 1, 3, 6, 9 e 12 Meses')
    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.savefig('forecast_plot.png')  # Salva o gráfico como forecast_plot.png
    st.image('forecast_plot.png')



# plot graph
#plot_xgboost(df_melted)



def save_plot2(df, descricao, forecast,model, intervalo):
    #st.write('plot prophet')
    #st.table(df)
    #st.table(forecast)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], label='Valores Atuais', color='blue', marker='o')
    plt.plot(forecast['ds'], forecast['yhat'], label='Valores Previstos', color='red', linestyle='--')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2)

    # Configurando o formato do eixo y para inteiro
    #plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
    
    plt.title('Valores Atuais e Previsões Ajustadas')
    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('prophetplot1.png')


    #fig1 = m.plot(forecast)
    #m.plot(forecast)
    #fig1.savefig('prophetplot1.png')
    #st.markdown("### Produto:")
    #st.subheader(descricao)
    #st.markdown('### Próximos: ' + str(intervalo) +' meses')
    #st.markdown('### Prophet')
    st.image('prophetplot1.png')
    #
    #st.markdown("### Components: Trend Weekly Yearly Daily")
    #fig2 =  m.plot_components(forecast)
    #fig2.savefig("prophetplot2.png")
    #st.image('prophetplot2.png')



def predict3(temp, intervalo):
  df = pd.DataFrame()
  df['ds'] = pd.to_datetime(temp['mes-ano'])  # Coluna de datas mensais
  df['y'] = temp['valor']  # Coluna de valores
  df = df.sort_values(by='ds')
 
  # Criar e treinar o modelo Prophet com sazonalidade anual
  model = Prophet(daily_seasonality=False, yearly_seasonality=True) #, seasonality_mode='additive') # intervalo de confiança igual a 80%, para alterar: interval_width=0.95
  #model.add_seasonality(name='annual', period=365.25, fourier_order=3)
  # Treinamento do modelo
  model.fit(df)
  
    
  # Criar datas futuras para a previsão (mensalmente)
  future_dates = model.make_future_dataframe(periods=intervalo, freq='M')  # 12 meses no futuro

  # Fazer previsões
  forecast = model.predict(future_dates)
    
  return (forecast,model, df)
           
def main():
 
 
    """B3 App """
     
    html_page = """
     <div style="background-color:tomato;padding=50px">
         <p style='text-align:center;font-size:50px;font-weight:bold'>Previsoes</p>
     </div>
               """
    st.markdown(html_page, unsafe_allow_html=True)
 
    #image = Image.open("logo.png")
    #st.sidebar.image(image,caption="", use_column_width=True)
    
    activities = ["Predictions","About"]
    choice = st.sidebar.radio("Home",activities)
    
    if choice == 'Predictions':
        #predict('TAEE4.SA') 
        st.markdown("### Escolha um produto / Descrição")
        descricao = st.selectbox('Descricao',descricoes, label_visibility = 'hidden')
        st.write('Código:', descricao_para_codigo[descricao])
        
        #symbol, description,forecast,model = predict3(option)
        temp = df_melted.loc[df_melted['Descrição']== descricao]
        st.write("Quantidade de Vendas:", temp.shape[0])
        st.table(temp)
        # Cria o gráfico e salva como 'atual.png'
        fig, ax = plt.subplots()
        temp.plot(x='mes-ano', y='valor', ax=ax)
        st.subheader(descricao)
        ax.set_title('Evolução dos Valores ao Longo do Tempo')
        plt.savefig('atual.png')  # Salva o gráfico

        # Exibe a imagem salva
        st.image('atual.png')
        
        st.markdown("### Próximos 12 meses")
        algoritmo = st.radio("Método  ",['Prophet', 'Xgboost',  'Ambos'], 
        captions=[
        "rápido",
        "demorado",
        "processa ambos métodos",
    ], horizontal = True)
        intervalo = 12
        
        if st.button("Executar a previsão"):
            try:
               #symbol, description,forecast,model = predict2(option)
               #save_plot(symbol, description,forecast,model)
               with st.spinner('Aguarde o processamento...treinamento do modelo e plot do gráfico'):
               
                   if algoritmo == 'Prophet':
                   
                       forecast,model, df = predict3(temp, intervalo)
                       st.markdown('### Prophet')
                       
                       st.table(forecast)
                       
                       save_plot2(df, descricao, forecast,model, intervalo)
                   
                   elif algoritmo == 'Xgboost':
                       
                       st.markdown('### Xgboost')
                       # Training model
                       #st.write('Training xgboost')
                       df_xgb, model = training_xgboost(temp)
                   
                       #st.write(model)
                       
                       # Exemplo de uso para prever os próximos 12 meses
                       predicoes_12_meses = forecast_xgboost(model, df_xgb, steps=12)
                       
                       st.table(predicoes_12_meses)
                       
                       plot_xgboost(df_xgb, predicoes_12_meses)
                   else:
                       forecast,model, df = predict3(temp, intervalo)
                       st.markdown('### Prophet')
                       save_plot2(df, descricao, forecast,model, intervalo)
                       
                       st.markdown('### Xgboost')
                       # Training model
                       #st.write('Training xgboost')
                       df_xgb, model = training_xgboost(temp)
                   
                       #st.write(model)
                       #st.table(df_xgb)
                       # Exemplo de uso para prever os próximos 12 meses
                       predicoes_12_meses = forecast_xgboost(model, df_xgb, steps=12)
                       plot_xgboost(df_xgb, predicoes_12_meses)
                       
                       
                                      
            except:
               st.write("Error descriçao: "+descricao)
               st.error('Checar.', icon="🚨")
 
         
    else:
        st.markdown('### Dados dos últimos 12 meses')
        
 
 
 
if __name__ == '__main__':
    main()



#!nohup streamlit run app.py &

#!streamlit run /content/app.py &>/content/logs.txt &

#ngrok.kill()

# Terminate ngrok port
#ngrok.kill()
# Set authentication (optional)
# Get your authentication token via https://dashboard.ngrok.com/auth
#
#NGROK_AUTH_TOKEN = "1gNjeFx7GPLDxTs5H60p5ZeZgl8_2NQZEtskwajdxd3ge8ZSx"
#ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Open an HTTPs tunnel on port 5000 for http://localhost:5000
#ngrok_tunnel = ngrok.connect(addr="8501", proto="http", bind_tls=True)
#print("Streamlit Tracking UI:", ngrok_tunnel.public_url)
