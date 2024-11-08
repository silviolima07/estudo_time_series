
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

import pmdarima as pm

import openpyxl
#file = 'Consumo x Previsão.xlsx'
#df = pd.read_excel(file)

# Carregar o arquivo XLS
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
    df['mes'] = pd.to_datetime(df['ano-mes']).dt.month
    df['ano'] = pd.to_datetime(df['ano-mes']).dt.year

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
    
    df['ano-mes'] = pd.to_datetime(df['ano-mes'], format='%Y-%m', errors='coerce')
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
    ultima_data = df['ano-mes'].iloc[-1]

    print('Ultima data:',ultima_data)
    # Criando um DataFrame para as previsões
    previsao_data = {
    'ano-mes': [
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
    df = df.dropna(subset=['ano-mes'])
    #st.table(df)

    # Exibindo o DataFrame atualizado
    #st.table(df)
    #st.table(df_previsao)

    # Plotando os valores históricos e as previsões
    plt.figure(figsize=(12, 6))
    plt.plot(df['ano-mes'], df['valor'], label='Valores Atuais', color='blue', marker='o')
    plt.axvline(x=ultima_data, color='green', linestyle='--', label='Início das Previsões')
    plt.plot(df_previsao['ano-mes'].iloc[-5:], df_previsao['valor'].iloc[-5:], color='red', label='Previsões', marker='o')
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



def predict3(temp, intervalo, sazonality):
  df = pd.DataFrame()
  df['ds'] = pd.to_datetime(temp['ano-mes'])  # Coluna de datas mensais
  df['y'] = temp['valor']  # Coluna de valores
  df = df.sort_values(by='ds')
 
  # Criar e treinar o modelo Prophet com sazonalidade anual
  model = Prophet(daily_seasonality=False, yearly_seasonality=sazonality) #, seasonality_mode='additive') # intervalo de confiança igual a 80%, para alterar: interval_width=0.95
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
        uploaded_files = st.file_uploader("Escolha até 4 arquivo XLS", type=["xlsx"], accept_multiple_files=True, key="xlsx_files")
        
        if uploaded_files:
            # Lista para armazenar cada DataFrame carregado
            dataframes = []

            # Loop para processar cada arquivo carregado
            for uploaded_file in uploaded_files[:4]:  # Limita a 4 arquivos
                df = pd.read_excel(uploaded_file)  # Lê o arquivo como um DataFrame
                dataframes.append(df)  # Adiciona o DataFrame à lista
                st.write(f"Exibindo dados do arquivo: {uploaded_file.name}")
                st.dataframe(df)  # Mostra o conteúdo do arquivo carregado

            # Concatena todos os DataFrames em um só
            consolidated_df = pd.concat(dataframes, ignore_index=True)
    
            st.write("Planilha Consolidada:")
            st.dataframe(consolidated_df)  # Mostra o DataFrame consolidado

        #if uploaded_file is not None:
            # Ler o arquivo XLS
            #df = pd.read_excel(uploaded_file)
            df = consolidated_df

            # Filtra as colunas de data
            date_columns = [col for col in df.columns if isinstance(col, datetime.datetime)]

            # Filtra as colunas de data
            date_columns = [col for col in df.columns if isinstance(col, datetime.datetime)]

            # Converte o DataFrame para o formato longo (long format) para reorganizar as colunas de data
            df_melted = df.melt(id_vars=[col for col in df.columns if col not in date_columns],
                    value_vars=date_columns,
                    var_name='ano-mes', value_name='valor')

            # Convert 'mes-ano' column to datetime objects
            df_melted['ano-mes'] = pd.to_datetime(df_melted['ano-mes'])

            # Now you can use .dt.strftime
            df_melted['ano-mes'] = df_melted['ano-mes'].dt.strftime('%Y-%m')

            df_melted.dropna(inplace=True)

            descricoes = df_melted['Descrição'].to_list()
            codigos    = df_melted['Código'].to_list()
            #st.write('Criar dicionario descricao:codigo')
            # Cria o dicionário
            descricao_para_codigo = dict(zip(descricoes, codigos))
    
            st.markdown("### Escolha um produto / Descrição")
            descricao = st.selectbox('Descricao',descricoes, label_visibility = 'hidden')
            st.write('Código:', descricao_para_codigo[descricao])
        
            #symbol, description,forecast,model = predict3(option)
            temp = df_melted.loc[df_melted['Descrição']== descricao]
            st.write("Quantidade de Vendas:", temp.shape[0])
            
            st.markdown("### Tabela Original desse produto")
            st.table(temp)
            
            # Filtrar linhas com valor 0 e negativo
            
            
            
            temp_fil_maior_zero = temp.loc[temp.valor >0]
            temp_fil_maior_igual_zero = temp.loc[temp.valor>=0]
            
            st.markdown("#### Filtrar valores negativos e zero ou apenas valores abaixo de zero")
            filtro_zero = st.radio(" ",['Original', 'Valores a partir do zero', 'Valor acima de zero'], horizontal = True, label_visibility='hidden')
            
            if filtro_zero == 'Valores a partir do zero':
                temp = temp_fil_maior_igual_zero
                st.markdown("### Tabela filtrada - valores >= zero")
                st.write("Quantidade de Vendas:", temp.shape[0])
                st.table(temp)
                
            elif filtro_zero == 'Valor acima de zero':
                temp = temp_fil_maior_zero
                st.markdown("### Tabela filtrada - valores > zero")
                st.write("Quantidade de Vendas:", temp.shape[0])
                st.table(temp)
            else:
                st.markdown('### Tabela original')
                st.write("Quantidade de Vendas:", temp.shape[0])
                st.table(temp)
                             
              
            
            #st.markdown("### Tabela filtrada")
            #st.markdown("##### - Removidos valores 0 e negativos")
            #st.write("Quantidade de Vendas:", temp_fil_maior_zero.shape[0])
            #st.table(temp_fil_maior_zero)
            
            
            #filtro = st.radio("Original / Filtrada",['Original', 'Filtrada'], horizontal = True)
            
            #if filtro == 'Filtrada':
            #    temp = temp_fil_maior_zero
            #    st.markdown("Tabela filtrada")
            #else:
            #    st.markdown("Tabela original")            
            #df['data'] = pd.to_datetime(temp['mes-ano'], format='%Y-%m')
            #df = df.sort_values(by='data')
            #df['valor'] = temp['mes-ano']
            # Plotar o gráfico
            #plt.plot(df['data'], df['valor'])
            #plt.xlabel('Data')
            plt.ylabel('Valor')
            #plt.title('Exemplo de Gráfico com Datas Ordenadas')
            #plt.xticks(rotation=45)  # Rotacionar para facilitar a leitura das datas
            #plt.show()
            # Cria o gráfico e salva como 'atual.png'
            fig, ax = plt.subplots()
            temp = temp.sort_values(by='ano-mes')
            temp.plot(x='ano-mes', y='valor', ax=ax)
            st.subheader(descricao)
            st.write("Qtd:", temp.shape[0])
            #ax.set_title('Evolução dos Valores ao Longo do Tempo')
            
            plt.savefig('atual.png')  # Salva o gráfico

            # Exibe a imagem salva
            st.image('atual.png')
        
            st.markdown("### Próximos 12 meses")
            algoritmo = st.radio("Método  ",['Prophet', 'Xgboost',  'Arima'], 
        captions=[
        "rápido",
        "demorado",
        "Auto_arima",
    ], horizontal = True)
            intervalo = 12
        
            if algoritmo == 'Prophet':
                sazonality = st.radio("Considerar SAZONALIDADE  ",[True, False], horizontal = True)
        
            if st.button("Executar a previsão"):
                try:
                   #symbol, description,forecast,model = predict2(option)
                   #save_plot(symbol, description,forecast,model)
                   with st.spinner('Aguarde o processamento...treinamento do modelo e plot do gráfico'):
               
                       if algoritmo == 'Prophet':
                                     
                           forecast,model, df = predict3(temp, intervalo, sazonality)
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
                       #forecast,model, df = predict3(temp, intervalo)
                       #st.markdown('### Prophet')
                       #save_plot2(df, descricao, forecast,model, intervalo)
                       
                       #st.markdown('### Xgboost')
                       # Training model
                       #st.write('Training xgboost')
                       #df_xgb, model = training_xgboost(temp)
                   
                       #st.write(model)
                       #st.table(df_xgb)
                       # Exemplo de uso para prever os próximos 12 meses
                       #predicoes_12_meses = forecast_xgboost(model, df_xgb, steps=12)
                       #plot_xgboost(df_xgb, predicoes_12_meses)
                       # Inicializar o Auto-ARIMA
                           st.markdown('### Arima')
                           # Training model
                           df = temp[['ano-mes','valor']]
                           df['ano-mes'] = pd.to_datetime(temp['ano-mes'])
                       
                           df.set_index('ano-mes', inplace=True)
                       
                      
                           model = pm.auto_arima(df['valor'], seasonal=True, m=3, trace=True, error_action='ignore', suppress_warnings=True)
                           
                           
                           # Fazer previsões para os próximos 12 meses
                           n_periods = 12
                           forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

                           # Criar um DataFrame com os valores previstos
                           future_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')
                           forecast_df = pd.DataFrame({'Previsão': forecast}, index=future_dates)
                       
                           st.table(forecast_df)
                           plt.figure(figsize=(10, 6))
                           plt.plot(df.index, df['valor'], label='Valores Reais')
                           plt.plot(forecast_df.index, forecast_df['Previsão'], label='Previsão Auto-ARIMA', color='orange')
                           plt.fill_between(forecast_df.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='Intervalo de Confiança')
                           plt.legend()
                           plt.title('Previsão de Vendas - Auto-ARIMA')
                           #plt.xlabel('Data')
                           #plt.ylabel('Vendas')
                           plt.savefig("arima.png")
                           st.image('arima.png')
                       
                                      
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
