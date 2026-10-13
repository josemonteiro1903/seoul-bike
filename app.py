# Import de bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

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
st.header("🧪 7. Teste de hipótese: chuva e demanda")

st.markdown("""
**Objetivo:** investigar se a demanda diária por bicicletas compartilhadas
difere entre dias chuvosos e dias sem chuva.

- **H₀:** a média da demanda diária é igual nos dois grupos.
- **H₁:** a média da demanda diária é diferente nos dois grupos.
- **Nível de significância:** 5% (α = 0,05).
- **Teste:** teste t de Welch para duas amostras independentes.
- **Intervalo de confiança:** 95% para a diferença entre as médias.
""")

st.caption(
    "São considerados dias chuvosos aqueles com precipitação total diária "
    "maior que zero. A análise considera somente dias de funcionamento."
)

# Preparação dos dados diários
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
    # Estatísticas descritivas
    media_chuva = grupo_chuva.mean()
    media_sem_chuva = grupo_sem_chuva.mean()
    diferenca = media_chuva - media_sem_chuva

    n_chuva = len(grupo_chuva)
    n_sem_chuva = len(grupo_sem_chuva)

    variancia_chuva = grupo_chuva.var(ddof=1)
    variancia_sem_chuva = grupo_sem_chuva.var(ddof=1)

    # Teste t de Welch
    resultado = stats.ttest_ind(
        grupo_chuva,
        grupo_sem_chuva,
        equal_var=False,
        alternative="two-sided"
    )

    valor_p = resultado.pvalue
    estatistica_t = resultado.statistic

    # Erro-padrão e graus de liberdade de Welch
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

    # Intervalo de confiança de 95%
    t_critico = stats.t.ppf(
        0.975,
        df=graus_liberdade
    )

    limite_inferior = diferenca - t_critico * erro_padrao
    limite_superior = diferenca + t_critico * erro_padrao

    # Indicadores
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

    if valor_p < alpha:
        st.success(
            "Rejeitamos H₀ ao nível de 5%. Há evidência estatística "
            "de diferença entre as médias de demanda diária dos dois grupos."
        )
    else:
        st.info(
            "Não rejeitamos H₀ ao nível de 5%. Os dados não fornecem "
            "evidência estatística suficiente para afirmar que as médias "
            "de demanda diária diferem."
        )

    if limite_inferior > 0:
        st.write(
            "O intervalo de confiança sugere maior demanda média "
            "nos dias chuvosos."
        )
    elif limite_superior < 0:
        st.write(
            "O intervalo de confiança sugere menor demanda média "
            "nos dias chuvosos."
        )
    else:
        st.write(
            "O intervalo de confiança inclui zero, indicando que "
            "a diferença entre as médias é compatível com ausência "
            "de diferença estatística ao nível de 5%."
        )

    st.caption(
        "Limitação: o teste pressupõe independência entre as observações "
        "diárias e usa uma aproximação para a distribuição das médias. "
        "Dias consecutivos podem apresentar dependência temporal; por isso, "
        "o resultado deve ser interpretado com cautela. Associação não "
        "comprova causalidade."
    )
st.divider()

# Tabela

st.subheader("Dados filtrados")
st.dataframe(df)