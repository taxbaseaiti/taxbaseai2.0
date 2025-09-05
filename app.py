import streamlit as st
import pandas as pd
from io import BytesIO
import dropbox
import openai
import altair as alt

st.write("Todos os secrets:", st.secrets)

# Carrega credenciais do Dropbox
dbx_cfg      = st.secrets["dropbox"]
ACCESS_TOKEN = dbx_cfg["access_token"]
BASE_PATH    = dbx_cfg["base_path"].rstrip("/")
dbx = dropbox.Dropbox(ACCESS_TOKEN)

# 1. Ingestão
@st.cache_data
def load_csv_from_dropbox(filename: str, expected_cols: list[str]) -> pd.DataFrame | None:
    path = f"{BASE_PATH}/{filename}"
    try:
        _, res = dbx.files_download(path=path)
    except dropbox.exceptions.ApiError as e:
        st.error(f"Não foi possível baixar {filename}: {e.error}")
        return None
    df = pd.read_csv(BytesIO(res.content))
    missing = set(expected_cols) - set(df.columns)
    if missing:
        st.error(f"Colunas faltando em {filename}: {missing}")
        return None
    return df

# 2. Normalização e Limpeza (unchanged)
COMMON_COLUMNS = {
    "nome_empresa":"company", "descrição":"account", "descricao":"account",
    "valor":"amount", "saldo_atual":"amount"
}
ACCOUNT_MAP = {
    # DRE – Receitas
    "RECEITA BRUTA DE VENDAS E MERCADORIAS":        "receita_operacional",
    "RECEITA DE PRESTAÇÃO DE SERVIÇOS":            "receita_operacional",
    "(-) IMPOSTOS SOBRE VENDAS E SERVIÇOS":        "impostos_operacionais",
    "RECEITA LÍQUIDA":                             "receita_liquida",

    # DRE – Custos
    "(-) MATERIAL APLICADO":                       "custo_material",
    "(-) DEPRECIAÇÕES, AMORTIZAÇÕES E EXAUSTÕES":  "custo_depreciacao",
    "(-) COMBUSTÍVEIS E ENERGIA ELÉTRICA":         "custo_energia",
    "(-) CUSTOS DOS PRODUTOS VENDIDOS":            "custo_vendas",
    "LUCRO BRUTO":                                 "lucro_bruto",

    # DRE – Despesas Operacionais
    "(-) DESPESAS OPERACIONAIS":                   "despesas_operacionais",
    "(-) DESPESAS COM PESSOAL (VENDAS)":           "despesa_pessoal_vendas",
    "(-) DESPESAS COM ENTREGA":                    "despesa_entrega",
    "(-) DESPESAS COM VIAGENS E REPRESENTAÇÕES":   "despesa_viagens",
    "(-) DESPESAS GERAIS (VENDAS)":                "despesa_vendas_geral",
    "(-) DESPESAS COM PESSOAL (ADMINISTRATIVAS)":  "despesa_pessoal_admin",
    "(-) IMPOSTOS, TAXAS E CONTRIBUIÇÕES":         "despesa_impostos_taxas",
    "(-) DESPESAS GERAIS (ADMINISTRATIVAS)":       "despesa_admin_geral",
    "(-) DESPESAS FINANCEIRAS":                    "despesa_financeira",

    # DRE – Receitas e Descontos Financeiros
    "JUROS E DESCONTOS":                           "receita_financeira",
    "RECEITAS DIVERSAS":                           "receita_diversas",

    # DRE – Resultados
    "RESULTADO OPERACIONAL":                       "resultado_operacional",
    "RESULTADO ANTES DO IR E CSL":                 "resultado_antes_ir_csll",
    "LUCRO LÍQUIDO DO EXERCÍCIO":                  "lucro_liquido",


    # BALANÇO – Ativo Circulante
    "ATIVO CIRCULANTE":                            "ativo_circulante",
    "DISPONÍVEL":                                  "disponivel",
    "BANCOS CONTA MOVIMENTO":                      "disponivel",
    "BANCO SICOOB":                                "disponivel",
    "CLIENTES":                                    "contas_receber",
    "DUPLICATAS A RECEBER":                        "contas_receber",
    "OUTROS CRÉDITOS":                             "outros_creditos",
    "TRIBUTOS A RECUPERAR/COMPENSAR":              "tributos_recuperar",
    "IPI A RECUPERAR":                             "tributos_recuperar",
    "ICMS A RECUPERAR":                            "tributos_recuperar",
    "COFINS A RECUPERAR":                          "tributos_recuperar",
    "PIS A RECUPERAR":                             "tributos_recuperar",
    "ESTOQUE":                                     "estoque",
    "MERCADORIAS, PRODUTOS E INSUMOS":             "estoque",
    "MERCADORIAS PARA REVENDA":                    "estoque",
    "MATÉRIA-PRIMA":                               "estoque",

    # BALANÇO – Ativo Não Circulante / Imobilizado
    "ATIVO NÃO-CIRCULANTE":                        "ativo_nao_circulante",
    "SÓCIOS, ADMINISTRADORES E PESSOAS LIGADA":    "conta_socios",
    "CONTA CORRENTE DE SÓCIOS":                    "conta_socios",
    "IMOBILIZADO":                                 "imobilizado",
    "IMÓVEIS":                                     "imobilizado",
    "TERRENOS":                                    "imobilizado",
    "MÓVEIS E UTENSÍLIOS":                         "imobilizado",
    "MÁQUINAS, EQUIPAMENTOS E FERRAMENTAS":        "imobilizado",
    "MÁQUINAS E EQUIPAMENTOS":                     "imobilizado",
    "VEÍCULOS":                                    "imobilizado",
    "OUTRAS IMOBILIZAÇÕES":                        "imobilizado",
    "COMPUTADORES E ACESSÓRIOS":                   "imobilizado",
    "IMOBILIZADO EM ANDAMENTO":                    "imobilizado",
    "MÁQUINAS E EQUIPAMENTOS (EM ANDAMENTO)":      "imobilizado",

    # BALANÇO – Depreciações Acumuladas
    "(-) DEPRECIAÇÕES, AMORT. E EXAUS. ACUMUL":     "deprec_acum",
    "(-) DEPRECIAÇÕES DE MÓVEIS E UTENSÍLIOS":     "deprec_acum",
    "(-) DEPRECIAÇÕES DE MÁQUINAS, EQUIP. FER":    "deprec_acum",
    "(-) DEPRECIAÇÕES DE VEÍCULOS":                "deprec_acum",
    "(-) DEPREC. COMPUTADORES E ACESSÓRIOS":       "deprec_acum",

    # BALANÇO – Passivo Circulante
    "PASSIVO CIRCULANTE":                          "passivo_circulante",
    "FORNECEDORES":                                "passivo_circulante",
    "OBRIGAÇÕES TRIBUTÁRIAS":                      "passivo_circulante",
    "IMPOSTOS E CONTRIBUIÇÕES A RECOLHER":         "passivo_circulante",
    "IPI A RECOLHER":                              "passivo_circulante",
    "IMPOSTO DE RENDA A RECOLHER":                 "passivo_circulante",
    "CONTRIBUIÇÃO SOCIAL A RECOLHER":              "passivo_circulante",
    "IRRF A RECOLHER":                             "passivo_circulante",
    "OBRIGAÇÕES TRABALHISTA E PREVIDENCIÁRIA":     "passivo_circulante",
    "OBRIGAÇÕES COM O PESSOAL":                    "passivo_circulante",
    "SALÁRIOS E ORDENADOS A PAGAR":                "passivo_circulante",
    "PRÓ-LABORE A PAGAR":                          "passivo_circulante",
    "PARTIC DE LUCROS A PAGAR":                    "passivo_circulante",
    "OBRIGAÇÕES SOCIAIS":                          "passivo_circulante",
    "INSS A RECOLHER":                             "passivo_circulante",
    "FGTS A RECOLHER":                             "passivo_circulante",
    "IRRF SOBRE SALÁRIOS":                         "passivo_circulante",
    "PROVISÕES":                                   "passivo_circulante",
    "PROVISÕES PARA FÉRIAS":                       "passivo_circulante",
    "INSS SOBRE PROVISÕES PARA FÉRIAS":            "passivo_circulante",
    "FGTS SOBRE PROVISÕES PARA FÉRIAS":            "passivo_circulante",
    "PIS SOBRE PROVISÕES PARA 13º SALÁRIO":        "passivo_circulante",
    "OUTRAS OBRIGAÇÕES":                           "passivo_circulante",
    "CONTAS A PAGAR":                              "passivo_circulante",
    "CARTÃO DE CRÉDITO SICOOB A PAGAR":            "passivo_circulante",
    "TRANSITÓRIA - CARTÃO DE CRÉDITO SICOOB A PAGAR":"passivo_circulante",
    "PASSIVO NÃO-CIRCULANTE":                      "passivo_nao_circulante",

    # BALANÇO – Patrimônio Líquido
    "LUCROS OU PREJUÍZOS ACUMULADOS":              "patrimonio_liquido",
    "LUCROS ACUMULADOS":                           "patrimonio_liquido",
    "(-) PREJUÍZOS ACUMULADOS":                    "patrimonio_liquido",
    "PATRIMÔNIO LÍQUIDO":                          "patrimonio_liquido",

    # Totais para validação
    "ATIVO":                                       "total_ativo",
    "PASSIVO":                                     "total_passivo"
}
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COMMON_COLUMNS).copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df

def add_metadata(df, stmt, ref_date, cid):
    df["statement"]  = stmt
    df["ref_date"]   = pd.to_datetime(ref_date)
    df["company_id"] = cid
    return df

def apply_account_mapping(df):
    df["account_std"] = df["account"].map(ACCOUNT_MAP).fillna("outros")
    return df

def clean_data(df):
    df = df.dropna(subset=["amount"])
    df = df[df["amount"] != 0]
    df = df.drop_duplicates()
    return df

# 3. Fluxo Streamlit
st.title("MVP IA Contábil — Etapa 2: Indicadores")

empresa_id = st.text_input("ID da Empresa", "CICLOMADE")
data_ref   = st.date_input("Data de Referência", value=pd.to_datetime("2024-12-31"))
date_str   = data_ref.strftime("%Y-%m-%d")

dre_raw = load_csv_from_dropbox(f"DRE_{date_str}_{empresa_id}.csv",
                                ["nome_empresa","descrição","valor"])
bal_raw = load_csv_from_dropbox(f"BALANCO_{date_str}_{empresa_id}.csv",
                                ["nome_empresa","descrição","saldo_atual"])
if dre_raw is None or bal_raw is None:
    st.stop()

dre = clean_data(apply_account_mapping(
        add_metadata(standardize_columns(dre_raw),
                     "income_statement", data_ref, empresa_id)))
bal = clean_data(apply_account_mapping(
        add_metadata(standardize_columns(bal_raw),
                     "balance_sheet", data_ref, empresa_id)))

df_all = pd.concat([bal, dre], ignore_index=True)

# 4. Etapa 2 — Cálculo de Indicadores

# 4.1 Consolida DRE
dre_sum = dre.groupby("account_std")["amount"].sum()

receita_liq = dre_sum.get("receita_liquida", 0)
custo_vendas = dre_sum.get("custo_vendas", 0)
lucro_bruto = receita_liq - custo_vendas
despesas_op = dre_sum.get("despesas_operacionais", 0)
ebitda = lucro_bruto - despesas_op
lucro_liq = dre_sum.get("lucro_liquido", 0)

# 4.2 Consolida Balanço
bal_sum = bal.groupby("account_std")["amount"].sum()

ativo_circ = bal_sum.get("ativo_circulante", 0)
ativo_nc   = bal_sum.get("ativo_nao_circulante", 0)
pass_circ  = bal_sum.get("passivo_circulante", 0)
pass_nc    = bal_sum.get("passivo_nao_circulante", 0)
pat_liq    = bal_sum.get("patrimonio_liquido", 0)
estoque    = bal_sum.get("estoque", 0)

total_ativo = ativo_circ + ativo_nc
total_pass  = pass_circ + pass_nc
bal_valida = abs(total_ativo - (total_pass + pat_liq)) < 1e-2

# 4.3 Principais índices
liquidez_corrente = ativo_circ / pass_circ if pass_circ else None
liquidez_seca    = (ativo_circ - estoque) / pass_circ if pass_circ else None
endividamento    = total_pass / total_ativo if total_ativo else None
roa              = lucro_liq / total_ativo if total_ativo else None
roe              = lucro_liq / pat_liq if pat_liq else None

# 5. Exibição no Streamlit

st.header("Indicadores do DRE")
st.metric("Lucro Bruto", f"R$ {lucro_bruto:,.2f}")
st.metric("EBITDA",       f"R$ {ebitda:,.2f}")
st.metric("Lucro Líquido",f"R$ {lucro_liq:,.2f}")

st.header("Indicadores do Balanço")
st.metric("Liquidez Corrente", f"{liquidez_corrente:.2f}")
st.metric("Liquidez Seca",     f"{liquidez_seca:.2f}")
st.metric("Endividamento",     f"{endividamento:.2%}")
st.write(f"Ativo = Passivo + PL? {'✅' if bal_valida else '❌'}")

st.header("Rentabilidades")
st.metric("ROA (Lucro/Ativo)", f"{roa:.2%}")
st.metric("ROE (Lucro/PL)",    f"{roe:.2%}")

# 6. Exportação (opcional CSV/Excel)
report = pd.DataFrame({
    "Indicador": ["Lucro Bruto","EBITDA","Lucro Líquido",
                  "Liquidez Corrente","Liquidez Seca","Endividamento",
                  "ROA","ROE"],
    "Valor":     [lucro_bruto, ebitda, lucro_liq,
                  liquidez_corrente, liquidez_seca, endividamento,
                  roa, roe]
})
csv = report.to_csv(index=False).encode("utf-8")
st.download_button("📥 Baixar Indicadores (CSV)", csv, "indicadores.csv")

# --- 7. Módulo de Perguntas & Respostas com IA ----------------------

# 7.1 Carrega chave da API
openai.api_key = st.secrets["openai"]["api_key"]

# 7.2 Templates de prompt (permanece igual)
PROMPT_TEMPLATES = {
    "trend_ebitda": 'Mostre a evolução do EBITDA considerando o valor atual de EBITDA ({ebitda:.2f}).',
    "liquidity_date": 'Qual a liquidez corrente em {date} considerando ativo circulante ({ativo_circ:.2f}) e passivo circulante ({pass_circ:.2f})?'
}

def format_indicators_table(df: pd.DataFrame) -> str:
    """Gera um mini-relatório textual dos indicadores."""
    lines = []
    for _, row in df.iterrows():
        val = row["Valor"]
        if isinstance(val, float):
            lines.append(f"- {row['Indicador']}: {val:,.2f}")
        else:
            lines.append(f"- {row['Indicador']}: {val}")
    return "\n".join(lines)

def ask_question(question: str, report_df: pd.DataFrame) -> str:
    """
    Recebe uma pergunta e o DataFrame de indicadores.
    Tenta usar openai.chat.completions (SDK >=1.0.0);
    em fallback, usa openai.ChatCompletion (SDK <1.0.0).
    """
    tabela = format_indicators_table(report_df)
    prompt = f"""
Você é um assistente contábil. Seguem os indicadores calculados na data de referência:

{tabela}

Pergunta: {question}

Forneça uma resposta objetiva e fundamentada nos números acima.
"""
    try:
        # primeiro, tentamos a interface v1+ (openai>=1.0.0)
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente contábil especializado em indicadores financeiros."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.0,
            max_tokens=250
        )
        return completion.choices[0].message.content.strip()
    except AttributeError:
        # fallback para versões antigas (openai<1.0.0)
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um assistente contábil especializado em indicadores financeiros."},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.0,
                max_tokens=250
            )
            return completion.choices[0].message.content.strip()
        except Exception as e2:
            return f"🚨 Erro na interface antiga do SDK OpenAI: {e2}"
    except Exception as e:
        return f"🚨 Erro ao consultar IA: {e}"

# --- 8. Interface Interativa com Navegação por Páginas ---

# configurações no sidebar
st.sidebar.title("📊 Navegação")
page = st.sidebar.radio("Escolha a página", ["Visão Geral", "Dashboards", "Chatbot"])

# Visão Geral: resumo e botão de refresh
if page == "Visão Geral":
    st.header("🏁 Visão Geral")
    st.markdown(f"**Empresa:** {empresa_id}   \n**Data:** {date_str}")

    if st.sidebar.button("🔄 Atualizar Indicadores"):
        st.experimental_rerun()

    st.subheader("Principais Métricas")
    cols = st.columns(3)
    cols[0].metric("Lucro Bruto",     f"R$ {lucro_bruto:,.2f}")
    cols[1].metric("EBITDA",           f"R$ {ebitda:,.2f}")
    cols[2].metric("Lucro Líquido",    f"R$ {lucro_liq:,.2f}")

    cols2 = st.columns(3)
    cols2[0].metric("Liquidez Corrente", f"{liquidez_corrente:.2f}")
    cols2[1].metric("Endividamento",     f"{endividamento:.2%}")
    cols2[2].metric("ROE",               f"{roe:.2%}")

    st.markdown("---")
    st.write("Tabela completa de indicadores")
    st.dataframe(report.style.format({"Valor":"R$ {:,.2f}"}))

# Dashboards: gráficos e tabelas filtráveis
elif page == "Dashboards":
    st.header("📈 Dashboards")

    # filtro por tipo de demonstração
    stmt = st.selectbox("Selecione demonstração", ["income_statement","balance_sheet"])
    df_view = df_all[df_all["statement"] == stmt]

    # filtro por conta-padrão
    accounts = sorted(df_view["account_std"].unique())
    sel = st.multiselect("Filtrar contas", accounts, default=accounts)
    df_filt = df_view[df_view["account_std"].isin(sel)]

    st.subheader("Tabela Filtrada")
    st.dataframe(df_filt[["account_std","amount"]])

    st.subheader("Gráfico de Barras por Conta")
    bar = (
        alt.Chart(df_filt.groupby("account_std")["amount"].sum().reset_index())
        .mark_bar()
        .encode(x=alt.X("account_std:N", sort="-y"), y="amount:Q", tooltip=["account_std","amount"])
        .properties(width=700)
    )
    st.altair_chart(bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Evolução Temporal (exemplo único ponto)")
    # caso tenha vários períodos, carregue vários df_all e plote linha
    st.write("Para ver evolução, carregue múltiplas datas e gere time-series.")
# Chatbot: área de texto e respostas da IA
else:
    st.header("🤖 Chatbot Contábil")
    pergunta = st.text_area("Digite sua pergunta sobre os indicadores acima")
    if st.button("Enviar"):
        if not pergunta:
            st.error("Insira sua pergunta antes de enviar.")
        else:
            with st.spinner("🔍 Consultando IA…"):
                resposta = ask_question(pergunta, report)
            st.markdown(resposta)