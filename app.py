# Import de bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuração

st.set_page_config(
    page_title="Seoul Bike",
    layout="wide"
)

st.title("📊 Mobilidade urbana e condições climáticas")

# Leitura dos dados

df = pd.read_csv(
    "data/SeoulBikeData.csv",
    encoding='unicode_escape'
)

st.sidebar.header("Filtros")

estacoes = sorted(df["Seasons"].dropna().unique())
estacoes_selecionadas = st.sidebar.multiselect(
    "Estação do ano",
    estacoes,
    default=estacoes
)

feriados = sorted(df["Holiday"].dropna().unique())
feriados_selecionados = st.sidebar.multiselect(
    "Feriado",
    feriados,
    default=feriados
)

dias = sorted(df["Functioning Day"].dropna().unique())
dias_selecionados = st.sidebar.multiselect(
    "Dia de funcionamento",
    dias,
    default=dias
)

df_filtrado = df[
    df["Seasons"].isin(estacoes_selecionadas)
    & df["Holiday"].isin(feriados_selecionados)
    & df["Functioning Day"].isin(dias_selecionados)
].copy()

if df_filtrado.empty:
    st.warning("Nenhum registro encontrado. Ajuste os filtros.")
    st.stop()

total_registros = len(df_filtrado)
total_alugueis = df_filtrado["Rented Bike Count"].sum()
media_alugueis = df_filtrado["Rented Bike Count"].mean()
media_temperatura = df_filtrado["Temperature(°C)"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros analisados", f"{total_registros:,}".replace(",", "."))
c2.metric("Total de aluguéis", f"{total_alugueis:,.0f}".replace(",", "."))
c3.metric("Média de aluguéis por hora", f"{media_alugueis:,.1f}".replace(",", "."))
c4.metric("Temperatura média", f"{media_temperatura:.1f} °C")

st.divider()

st.subheader("1. Demanda por horário")

demanda_hora = (
    df_filtrado.groupby("Hour", as_index=False)["Rented Bike Count"]
    .mean()
)

fig1 = px.line(
    demanda_hora,
    x="Hour",
    y="Rented Bike Count",
    markers=True,
    labels={
        "Hour": "Hora do dia",
        "Rented Bike Count": "Média de bicicletas alugadas"
    }
)
fig1.update_layout(xaxis=dict(dtick=1))
st.plotly_chart(fig1, use_container_width=True)
st.caption("Objetivo: identificar os horários com maior e menor demanda média.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("2. Temperatura e demanda")

    fig2 = px.scatter(
        df_filtrado,
        x="Temperature(°C)",
        y="Rented Bike Count",
        color="Seasons",
        opacity=0.5,
        labels={
            "Temperature(°C)": "Temperatura (°C)",
            "Rented Bike Count": "Bicicletas alugadas",
            "Seasons": "Estação"
        }
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Objetivo: explorar a relação entre temperatura e quantidade de aluguéis.")

with col2:
    st.subheader("3. Demanda por estação do ano")

    fig3 = px.box(
        df_filtrado,
        x="Seasons",
        y="Rented Bike Count",
        color="Seasons",
        labels={
            "Seasons": "Estação do ano",
            "Rented Bike Count": "Bicicletas alugadas"
        }
    )

    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Objetivo: comparar distribuição, mediana e valores extremos entre estações.")

col3, col4 = st.columns(2)

with col3:
    st.subheader("4. Chuva e demanda")

    fig4 = px.scatter(
        df_filtrado,
        x="Rainfall(mm)",
        y="Rented Bike Count",
        opacity=0.5,
        labels={
            "Rainfall(mm)": "Precipitação (mm)",
            "Rented Bike Count": "Bicicletas alugadas"
        }
    )

    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Objetivo: investigar como a demanda varia com a precipitação.")

with col4:
    st.subheader("5. Correlação entre variáveis")

    colunas_corr = [
        "Rented Bike Count",
        "Hour",
        "Temperature(°C)",
        "Humidity(%)",
        "Wind speed (m/s)",
        "Visibility (10m)",
        "Dew point temperature(°C)",
        "Solar Radiation (MJ/m2)",
        "Rainfall(mm)",
        "Snowfall (cm)"
    ]

    correlacoes = df_filtrado[colunas_corr].corr()

    fig5 = px.imshow(
        correlacoes,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels={"color": "Correlação"}
    )

    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Objetivo: identificar associações lineares entre variáveis numéricas.")

st.divider()
st.header("6. Modelo de regressão linear")

st.write(
    "O modelo estima a quantidade de bicicletas alugadas usando "
    "informações climáticas, horário, calendário e condições de funcionamento."
)

colunas_modelo = [
    "Hour",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "Month",
    "Day of Week",
    "Seasons",
    "Holiday",
    "Functioning Day"
]

dados_modelo = df_filtrado[
    ["Rented Bike Count"] + colunas_modelo
].dropna().copy()

X = pd.get_dummies(
    dados_modelo[colunas_modelo],
    columns=["Seasons", "Holiday", "Functioning Day"],
    drop_first=True,
    dtype=int
)

y = dados_modelo["Rented Bike Count"]

if len(dados_modelo) < 30:
    st.warning("Há poucos registros para uma avaliação confiável do modelo.")
else:
    indices = np.arange(len(dados_modelo))
    treino_idx, teste_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    modelo = LinearRegression()
    modelo.fit(X.iloc[treino_idx], y.iloc[treino_idx])

    previsoes = modelo.predict(X.iloc[teste_idx])
    valores_reais = y.iloc[teste_idx].to_numpy()

    mae = mean_absolute_error(valores_reais, previsoes)
    rmse = np.sqrt(mean_squared_error(valores_reais, previsoes))
    r2 = r2_score(valores_reais, previsoes)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.2f}")
    m2.metric("RMSE", f"{rmse:.2f}")
    m3.metric("R²", f"{r2:.3f}")

    st.caption(
        "MAE e RMSE medem o erro das previsões em unidades de bicicletas. "
        "Quanto menores, melhor; o R² mede a proporção da variação explicada "
        "pelo modelo no conjunto de teste."
    )

    fig6 = go.Figure()

    fig6.add_trace(
        go.Scatter(
            x=valores_reais,
            y=previsoes,
            mode="markers",
            name="Previsões",
            marker=dict(opacity=0.5)
        )
    )

    minimo = min(valores_reais.min(), previsoes.min())
    maximo = max(valores_reais.max(), previsoes.max())

    fig6.add_trace(
        go.Scatter(
            x=[minimo, maximo],
            y=[minimo, maximo],
            mode="lines",
            name="Previsão ideal"
        )
    )

    fig6.update_layout(
        xaxis_title="Quantidade real de bicicletas",
        yaxis_title="Quantidade prevista de bicicletas",
        legend_title="Série"
    )

    st.plotly_chart(fig6, use_container_width=True)
    st.caption(
        "Objetivo: comparar previsões e valores observados. "
        "Pontos próximos da linha representam previsões mais precisas."
    )

    coeficientes = pd.DataFrame({
        "Variável": X.columns,
        "Coeficiente": modelo.coef_
    })

    coeficientes["Impacto absoluto"] = coeficientes["Coeficiente"].abs()
    coeficientes = coeficientes.sort_values(
        "Impacto absoluto",
        ascending=False
    )

    with st.expander("Consultar coeficientes do modelo"):
        st.dataframe(
            coeficientes.drop(columns="Impacto absoluto"),
            use_container_width=True,
            hide_index=True
        )
        st.caption(
            "Cada coeficiente representa a variação estimada na demanda "
            "associada a uma unidade da variável, mantendo as demais constantes."
        )