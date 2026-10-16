
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor



st.set_page_config(
    page_title="Seoul Bike",
    layout="wide"
)

st.title("📊 Mobilidade urbana e condições climáticas")



df = pd.read_csv(
    "data/SeoulBikeData.csv",
    encoding='unicode_escape'
)

df_tratado = df.copy()

df_tratado['Date'] = pd.to_datetime(df_tratado['Date'], format='%d/%m/%Y')
df_tratado['IsWeekend'] = df_tratado['Date'].dt.dayofweek.isin([5, 6]).astype(int)

df_tratado = df_tratado[df_tratado['Functioning Day'] == 'Yes'].copy()
df_tratado = df_tratado.drop(columns=['Functioning Day'])

df_tratado = pd.get_dummies(df_tratado, columns=['Seasons'], drop_first=True)

df_tratado['Holiday_Yes'] = (df_tratado['Holiday'] == 'Holiday').astype(int)
df_tratado = df_tratado.drop(columns=['Holiday'])

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
    "O R² mede a proporção da variação explicada pelo modelo no conjunto de teste."
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
    st.caption("Objetivo: comparar previsões e valores observados.")

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

        st.divider()
st.header("📈 Modelo de regressão linear")

colunas_modelo = [
    'Rented Bike Count',
    'Hour',
    'Temperature(°C)',
    'Humidity(%)',
    'Wind speed (m/s)',
    'Solar Radiation (MJ/m2)',
    'Rainfall(mm)',
    'Snowfall (cm)',
    'IsWeekend',
    'Seasons_Spring',
    'Seasons_Summer',
    'Seasons_Winter',
    'Holiday_Yes'
]

df_modelo = df_tratado[colunas_modelo].copy()

X = df_modelo.drop(columns=['Rented Bike Count'])
y = df_modelo['Rented Bike Count']
X = sm.add_constant(X)

modelo = sm.OLS(y, X).fit()

st.subheader("Desempenho do modelo")
c1, c2, c3 = st.columns(3)
c1.metric("R²", f"{modelo.rsquared:.3f}")
c2.metric("R² ajustado", f"{modelo.rsquared_adj:.3f}")
c3.metric("F-statistic", f"{modelo.fvalue:.1f}")

st.subheader("Coeficientes")
coeficientes = pd.DataFrame({
    'Variável': modelo.params.index,
    'Coeficiente': modelo.params.values,
    'P-valor': modelo.pvalues.values,
    'IC inferior': modelo.conf_int()[0].values,
    'IC superior': modelo.conf_int()[1].values
})
coeficientes['P-valor'] = coeficientes['P-valor'].apply(lambda x: f"{x:.5f}")
st.dataframe(coeficientes)

st.subheader("Coeficientes do modelo")
fig_coef = px.bar(
    coeficientes[coeficientes['Variável'] != 'const'],
    x='Coeficiente',
    y='Variável',
    orientation='h',
    color='Coeficiente',
    color_continuous_scale='RdBu_r',
    labels={'Coeficiente': 'Coeficiente', 'Variável': 'Variável'}
)
st.plotly_chart(fig_coef, use_container_width=True)

st.divider()
st.header("🧪 7. Teste de hipótese: chuva e demanda")

st.markdown("""
**Objetivo:** investigar se a demanda diária por bicicletas compartilhadas
difere entre dias chuvosos e dias sem chuva.
""")

st.caption(
    "São considerados dias chuvosos aqueles com precipitação total diária "
    "maior que zero. A análise considera somente dias de funcionamento."
)



dados_teste = df[
    (df["Functioning Day"] == "Yes")
    & (df["Seasons"].isin(estacoes_selecionadas))
    & (df["Holiday"].isin(feriados_selecionados))
].copy()

dados_diarios = (
    dados_teste.groupby("Date", as_index=False)
    .agg(
        Demanda_Diaria=("Rented Bike Count", "sum"),
        Chuva_Diaria=("Rainfall(mm)", "sum")
    )
)

dados_diarios["Grupo"] = np.where(
    dados_diarios["Chuva_Diaria"] > 0,
    "Dia chuvoso",
    "Dia sem chuva"
)

grupo_chuva = dados_diarios.loc[
    dados_diarios["Grupo"] == "Dia chuvoso",
    "Demanda_Diaria"
].dropna()

grupo_sem_chuva = dados_diarios.loc[
    dados_diarios["Grupo"] == "Dia sem chuva",
    "Demanda_Diaria"
].dropna()

if len(grupo_chuva) < 2 or len(grupo_sem_chuva) < 2:
    st.warning(
        "Não há dias suficientes em ambos os grupos para executar "
        "o teste. Altere os filtros e tente novamente."
    )
else:

    media_chuva = grupo_chuva.mean()
    media_sem_chuva = grupo_sem_chuva.mean()
    diferenca = media_chuva - media_sem_chuva

    n_chuva = len(grupo_chuva)
    n_sem_chuva = len(grupo_sem_chuva)

    variancia_chuva = grupo_chuva.var(ddof=1)
    variancia_sem_chuva = grupo_sem_chuva.var(ddof=1)


    resultado = stats.ttest_ind(
        grupo_chuva,
        grupo_sem_chuva,
        equal_var=False,
        alternative="two-sided"
    )

    valor_p = resultado.pvalue
    estatistica_t = resultado.statistic


    termo_chuva = variancia_chuva / n_chuva
    termo_sem_chuva = variancia_sem_chuva / n_sem_chuva

    erro_padrao = np.sqrt(termo_chuva + termo_sem_chuva)

    graus_liberdade = (
        (termo_chuva + termo_sem_chuva) ** 2
        / (
            termo_chuva ** 2 / (n_chuva - 1)
            + termo_sem_chuva ** 2 / (n_sem_chuva - 1)
        )
    )


    t_critico = stats.t.ppf(
        0.975,
        df=graus_liberdade
    )

    limite_inferior = diferenca - t_critico * erro_padrao
    limite_superior = diferenca + t_critico * erro_padrao


    a, b, c = st.columns(3)

    a.metric(
        "Média diária — dias chuvosos",
        f"{media_chuva:,.0f} bicicletas".replace(",", ".")
    )

    b.metric(
        "Média diária — dias sem chuva",
        f"{media_sem_chuva:,.0f} bicicletas".replace(",", ".")
    )

    c.metric(
        "Diferença entre médias",
        f"{diferenca:+,.0f} bicicletas".replace(",", ".")
    )

    st.subheader("Comparação da demanda diária")

    fig_teste = px.box(
        dados_diarios,
        x="Grupo",
        y="Demanda_Diaria",
        color="Grupo",
        points="outliers",
        labels={
            "Grupo": "Condição de chuva",
            "Demanda_Diaria": "Total de bicicletas alugadas por dia"
        }
    )

    st.plotly_chart(fig_teste, use_container_width=True)

    st.subheader("Resultado do teste estatístico")

    x, y, z = st.columns(3)

    x.metric("Estatística t", f"{estatistica_t:.3f}")
    y.metric("Valor-p", f"{valor_p:.5f}")
    z.metric("Graus de liberdade", f"{graus_liberdade:.1f}")

    st.markdown("**Intervalo de confiança de 95% para a diferença das médias**")

    st.write(
        f"Limite inferior: **{limite_inferior:,.2f} bicicletas/dia**"
        .replace(",", "X").replace(".", ",").replace("X", ".")
    )

    st.write(
        f"Limite superior: **{limite_superior:,.2f} bicicletas/dia**"
        .replace(",", "X").replace(".", ",").replace("X", ".")
    )

    st.caption(
        "A diferença é calculada como média dos dias chuvosos menos "
        "média dos dias sem chuva. Valores negativos indicam menor "
        "demanda média no grupo chuvoso."
    )

    st.subheader("Interpretação estatística")

    alpha = 0.05


st.divider()
st.header("🧪 8. Teste de hipótese: feriado e demanda")

st.markdown("""
**Objetivo:** investigar se a demanda média por hora difere entre feriados e dias normais.
""")

grupo_feriado = df_tratado.loc[df_tratado['Holiday_Yes'] == 1, 'Rented Bike Count'].dropna()
grupo_nao_feriado = df_tratado.loc[df_tratado['Holiday_Yes'] == 0, 'Rented Bike Count'].dropna()

if len(grupo_feriado) < 2 or len(grupo_nao_feriado) < 2:
    st.warning("Não há dados suficientes em ambos os grupos para executar o teste.")
else:
    media_feriado = grupo_feriado.mean()
    media_nao_feriado = grupo_nao_feriado.mean()
    diferenca = media_feriado - media_nao_feriado

    n_feriado = len(grupo_feriado)
    n_nao_feriado = len(grupo_nao_feriado)

    var_feriado = grupo_feriado.var(ddof=1)
    var_nao_feriado = grupo_nao_feriado.var(ddof=1)

    resultado = stats.ttest_ind(
        grupo_feriado,
        grupo_nao_feriado,
        equal_var=False,
        alternative='two-sided'
    )

    valor_p = resultado.pvalue
    estatistica_t = resultado.statistic

    termo_feriado = var_feriado / n_feriado
    termo_nao_feriado = var_nao_feriado / n_nao_feriado
    erro_padrao = np.sqrt(termo_feriado + termo_nao_feriado)

    graus_liberdade = (
        (termo_feriado + termo_nao_feriado) ** 2
        / (
            termo_feriado ** 2 / (n_feriado - 1)
            + termo_nao_feriado ** 2 / (n_nao_feriado - 1)
        )
    )

    t_critico = stats.t.ppf(0.975, df=graus_liberdade)
    limite_inferior = diferenca - t_critico * erro_padrao
    limite_superior = diferenca + t_critico * erro_padrao

    a, b, c = st.columns(3)
    a.metric("Média feriado", f"{media_feriado:,.0f}".replace(",", "."))
    b.metric("Média não feriado", f"{media_nao_feriado:,.0f}".replace(",", "."))
    c.metric("Diferença", f"{diferenca:+,.0f}".replace(",", "."))

    st.subheader("Comparação da demanda")

    fig_feriado = px.box(
        df_tratado,
        x='Holiday_Yes',
        y='Rented Bike Count',
        color='Holiday_Yes',
        labels={
            'Holiday_Yes': 'Feriado (1 = sim, 0 = não)',
            'Rented Bike Count': 'Bicicletas alugadas por hora'
        }
    )
    st.plotly_chart(fig_feriado, use_container_width=True)

    st.subheader("Resultado do teste estatístico")

    x, y, z = st.columns(3)
    x.metric("Estatística t", f"{estatistica_t:.3f}")
    y.metric("Valor-p", f"{valor_p:.5f}")
    z.metric("Graus de liberdade", f"{graus_liberdade:.1f}")

    st.markdown("**Intervalo de confiança de 95% para a diferença das médias**")
    st.write(f"Limite inferior: **{limite_inferior:,.2f} bicicletas/hora**")
    st.write(f"Limite superior: **{limite_superior:,.2f} bicicletas/hora**")

    st.divider()
st.header("🧪 9. Teste de hipótese: fim de semana e demanda")

st.markdown("""
**Objetivo:** investigar se a demanda média por hora difere entre fins de semana e dias úteis.
""")

grupo_fim_semana = df_tratado.loc[df_tratado['IsWeekend'] == 1, 'Rented Bike Count'].dropna()
grupo_dia_util = df_tratado.loc[df_tratado['IsWeekend'] == 0, 'Rented Bike Count'].dropna()

if len(grupo_fim_semana) < 2 or len(grupo_dia_util) < 2:
    st.warning("Não há dados suficientes em ambos os grupos para executar o teste.")
else:
    media_fim_semana = grupo_fim_semana.mean()
    media_dia_util = grupo_dia_util.mean()
    diferenca = media_fim_semana - media_dia_util

    n_fim_semana = len(grupo_fim_semana)
    n_dia_util = len(grupo_dia_util)

    var_fim_semana = grupo_fim_semana.var(ddof=1)
    var_dia_util = grupo_dia_util.var(ddof=1)

    resultado = stats.ttest_ind(
        grupo_fim_semana,
        grupo_dia_util,
        equal_var=False,
        alternative='two-sided'
    )

    valor_p = resultado.pvalue
    estatistica_t = resultado.statistic

    termo_fim_semana = var_fim_semana / n_fim_semana
    termo_dia_util = var_dia_util / n_dia_util
    erro_padrao = np.sqrt(termo_fim_semana + termo_dia_util)

    graus_liberdade = (
        (termo_fim_semana + termo_dia_util) ** 2
        / (
            termo_fim_semana ** 2 / (n_fim_semana - 1)
            + termo_dia_util ** 2 / (n_dia_util - 1)
        )
    )

    t_critico = stats.t.ppf(0.975, df=graus_liberdade)
    limite_inferior = diferenca - t_critico * erro_padrao
    limite_superior = diferenca + t_critico * erro_padrao

    a, b, c = st.columns(3)
    a.metric("Média fim de semana", f"{media_fim_semana:,.0f}".replace(",", "."))
    b.metric("Média dia útil", f"{media_dia_util:,.0f}".replace(",", "."))
    c.metric("Diferença", f"{diferenca:+,.0f}".replace(",", "."))

    st.subheader("Comparação da demanda")

    fig_fim_semana = px.box(
        df_tratado,
        x='IsWeekend',
        y='Rented Bike Count',
        color='IsWeekend',
        labels={
            'IsWeekend': 'Fim de semana (1 = sim, 0 = não)',
            'Rented Bike Count': 'Bicicletas alugadas por hora'
        }
    )
    st.plotly_chart(fig_fim_semana, use_container_width=True)

    st.subheader("Resultado do teste estatístico")

    x, y, z = st.columns(3)
    x.metric("Estatística t", f"{estatistica_t:.3f}")
    y.metric("Valor-p", f"{valor_p:.5f}")
    z.metric("Graus de liberdade", f"{graus_liberdade:.1f}")

    st.markdown("**Intervalo de confiança de 95% para a diferença das médias**")
    st.write(f"Limite inferior: **{limite_inferior:,.2f} bicicletas/hora**")
    st.write(f"Limite superior: **{limite_superior:,.2f} bicicletas/hora**")

# Tabela

st.subheader("Dados filtrados")
st.dataframe(df_filtrado)