import streamlit as st
import streamlit_authenticator as stauth
import bcrypt
import pandas as pd
import numpy as np
from io import BytesIO
import dropbox
from openai import OpenAI
import altair as alt
import plotly.express as px
import faiss
import pickle
import time

# -----------------------------------------------------------------------------  
# 0. Settings & Secrets  
# -----------------------------------------------------------------------------
st.set_page_config(page_title="TaxbaseAI - Sua AI Contábil", layout="wide")

dbx_cfg      = st.secrets["dropbox"]
dbx          = dropbox.Dropbox(
    app_key=dbx_cfg["app_key"],
    app_secret=dbx_cfg["app_secret"],
    oauth2_refresh_token=dbx_cfg["refresh_token"]
)
BASE_PATH    = dbx_cfg["base_path"].rstrip("/")

client = OpenAI(api_key=st.secrets["openai"]["api_key"])


# -----------------------------------------------------------------------------  
# 0.1 Credentials & Authentication  
# -----------------------------------------------------------------------------
USERS = {
    "alice": {"name":"Alice Souza","password":bcrypt.hashpw("senhaAlice".encode(), bcrypt.gensalt()).decode(),"empresa":"JJMAX","role":"user"},
    "bob":   {"name":"Bob Oliveira","password":bcrypt.hashpw("senhaBob".encode(),   bcrypt.gensalt()).decode(),"empresa":"CICLOMADE","role":"user"},
    "admin": {"name":"Administrador","password":bcrypt.hashpw("senhaAdmin".encode(), bcrypt.gensalt()).decode(),"empresa":None,"role":"admin"}
}

credentials = {
    "usernames": {
        user: {"name":info["name"], "password":info["password"]}
        for user, info in USERS.items()
    }
}

cfg = st.secrets["auth"]
authenticator = stauth.Authenticate(
    credentials,
    cfg["cookie_name"],
    cfg["key"],
    cfg["expiry_days"],
)

# ======== LOGIN ========
if not st.session_state.get("logged_in"):
    authenticator.login(
        location="main",
        fields={
            "Form name": "Acesso ao Sistema",
            "Username": "Usuário",
            "Password": "Senha",
            "Login": "Entrar"
        }
    )

    name = st.session_state.get("name")
    authentication_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")

    if authentication_status:
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        # Define página inicial após login
        st.session_state["page"] = "Visão Geral"
        st.success(f"Bem-vindo, {name}!")
        st.experimental_rerun()

    elif authentication_status is False:
        st.error("Usuário ou senha incorretos")

    else:
        st.warning("Por favor, insira suas credenciais")

# -----------------------------------------------------------------------------  
# 1. FAISS Embeddings  
# -----------------------------------------------------------------------------
EMBED_INDEX_PATH = "embeddings.index"
META_PATH        = "embeddings_meta.pkl"
EMB_DIM          = 1536

def build_or_load_index():
    try:
        index = faiss.read_index(EMBED_INDEX_PATH)
        meta  = pickle.load(open(META_PATH, "rb"))
    except:
        index = faiss.IndexFlatL2(EMB_DIM)
        meta  = []
    return index, meta

def persist_index(index, meta):
    faiss.write_index(index, EMBED_INDEX_PATH)
    pickle.dump(meta, open(META_PATH, "wb"))

def semantic_search(query: str, index, meta, top_k=5):
    q_emb = client.embeddings.create(model="text-embedding-ada-002", input=[query]).data[0].embedding
    D, I  = index.search(np.array([q_emb], dtype="float32"), top_k)
    return [meta[i] for i in I[0] if 0 <= i < len(meta)]

def upsert_embedding(question: str, answer: str, index, meta):
    emb = client.embeddings.create(
        model="text-embedding-ada-002", input=[question+" ||| "+answer]
    ).data[0].embedding
    index.add(np.array([emb], dtype="float32"))
    meta.append({"q": question, "a": answer})
    persist_index(index, meta)

index, meta = build_or_load_index()

# -----------------------------------------------------------------------------  
# 2. Ingestão & Normalização & Mapeamento  
# -----------------------------------------------------------------------------
COMMON_COLUMNS = {
    "nome_empresa":"company", "descrição":"account", "descricao":"account",
    "valor":"amount", "saldo_atual":"amount"
}

ACCOUNT_MAP = {
        # demonstração de resultados (DRE)
    "RECEITA BRUTA DE VENDAS E MERCADORIAS":        "gross_revenue",
    "RECEITA DE PRESTAÇÃO DE SERVIÇOS":            "service_revenue",
    "(-) IMPOSTOS SOBRE VENDAS E SERVIÇOS":        "taxes_on_sales",
    "RECEITA LÍQUIDA":                             "net_revenue",
    "(-) MATERIAL APLICADO":                       "materials_applied",
    "(-) DEPRECIAÇÕES, AMORTIZAÇÕES E EXAUSTÕES":  "depreciation_amortization",
    "(-) COMBUSTÍVEIS E ENERGIA ELÉTRICA":         "fuel_and_energy",
    "(-) CUSTOS DOS PRODUTOS VENDIDOS":            "cogs",
    "LUCRO BRUTO":                                 "gross_profit",
    "(-) DESPESAS OPERACIONAIS":                   "operating_expenses",
    "(-) DESPESAS COM PESSOAL (VENDAS)":           "personnel_expenses_sales",
    "(-) DESPESAS COM ENTREGA":                    "delivery_expenses",
    "(-) DESPESAS COM VIAGENS E REPRESENTAÇÕES":   "travel_and_representation_expenses",
    "(-) DESPESAS GERAIS (VENDAS)":                "general_sales_expenses",
    "(-) DESPESAS COM PESSOAL (ADMINISTRATIVAS)":  "personnel_expenses_admin",
    "(-) IMPOSTOS, TAXAS E CONTRIBUIÇÕES":         "taxes_fees_contributions",
    "(-) DESPESAS GERAIS (ADMINISTRATIVAS)":       "general_admin_expenses",
    "(-) DESPESAS FINANCEIRAS":                    "financial_expenses",
    "JUROS E DESCONTOS":                           "interest_and_discounts",
    "RECEITAS DIVERSAS":                           "other_income",
    "RESULTADO OPERACIONAL":                       "operating_result",
    "RESULTADO ANTES DO IR E CSL":                 "ebit_before_tax_and_social",
    "LUCRO LÍQUIDO DO EXERCÍCIO":                  "net_income",
    
    # balanço patrimonial (ativo)
    "ATIVO CIRCULANTE":                            "current_assets",
    "DISPONÍVEL":                                  "cash",
    "BANCOS CONTA MOVIMENTO":                      "bank_accounts",
    "BANCO SICOOB":                                "bank_sicoob",
    "CLIENTES":                                    "accounts_receivable",
    "DUPLICATAS A RECEBER":                        "accounts_receivable",
    "OUTROS CRÉDITOS":                             "other_receivables",
    "TRIBUTOS A RECUPERAR/COMPENSAR":              "taxes_recoverable",
    "IPI A RECUPERAR":                             "taxes_recoverable_ipi",
    "ICMS A RECUPERAR":                            "taxes_recoverable_icms",
    "COFINS A RECUPERAR":                          "taxes_recoverable_cofins",
    "PIS A RECUPERAR":                             "taxes_recoverable_pis",
    "ESTOQUE":                                     "inventory",
    "MERCADORIAS, PRODUTOS E INSUMOS":             "inventory_goods_and_supplies",
    "MERCADORIAS PARA REVENDA":                    "inventory_for_resale",
    "MATÉRIA-PRIMA":                                "inventory_raw_materials",
    "ATIVO NÃO-CIRCULANTE":                        "non_current_assets",
    "SÓCIOS, ADMINISTRADORES E PESSOAS LIGADA":    "related_party_receivables",
    "CONTA CORRENTE DE SÓCIOS":                    "shareholder_current_accounts",
    "IMOBILIZADO":                                 "fixed_assets",
    "IMÓVEIS":                                     "fixed_assets_properties",
    "TERRENOS":                                    "fixed_assets_land",
    "MÓVEIS E UTENSÍLIOS":                         "fixed_assets_furniture_and_fixtures",
    "MÁQUINAS, EQUIPAMENTOS E FERRAMENTAS":        "fixed_assets_machinery_and_tools",
    "MÁQUINAS E EQUIPAMENTOS":                     "fixed_assets_machinery_and_equipment",
    "VEÍCULOS":                                    "fixed_assets_vehicles",
    "OUTRAS IMOBILIZACOES":                        "fixed_assets_other",
    "COMPUTADORES E ACESSORIOS":                   "fixed_assets_computers_and_accessories",
    "IMOBILIZADO EM ANDAMENTO":                    "fixed_assets_in_progress",
    "MÁQUINAS E EQUIPAMENTOS (EM ANDAMENTO)":      "fixed_assets_machinery_in_progress",
    "(-) DEPRECIAÇÕES, AMORT. E EXAUS. ACUMUL":     "accumulated_depreciation",
    "(-) DEPRECIAÇÕES DE MÓVEIS E UTENSÍLIOS":     "accumulated_depr_furniture_and_fixtures",
    "(-) DEPRECIAÇÕES DE MÁQUINAS, EQUIP. FER":    "accumulated_depr_machinery",
    "(-) DEPRECIAÇÕES DE VEÍCULOS":                "accumulated_depr_vehicles",
    "(-) DEPREC. COMPUTADORES E ACESSORIOS":       "accumulated_depr_computers",
    
    # balanço patrimonial (passivo)
    "PASSIVO CIRCULANTE":                          "current_liabilities",
    "FORNECEDORES":                                "accounts_payable",
    "OBRIGAÇÕES TRIBUTÁRIAS":                      "tax_liabilities",
    "IMPOSTOS E CONTRIBUIÇÕES A RECOLHER":         "tax_liabilities",
    "IPI A RECOLHER":                              "tax_liabilities_ipi",
    "IMPOSTO DE RENDA A RECOLHER":                 "tax_liabilities_income_tax",
    "CONTRIBUIÇÃO SOCIAL A RECOLHER":              "tax_liabilities_social_contribution",
    "IRRF A RECOLHER":                             "tax_liabilities_irrf",
    "OBRIGAÇÕES TRABALHISTA E PREVIDENCIÁRIA":     "labor_and_social_liabilities",
    "OBRIGAÇÕES COM O PESSOAL":                    "personnel_liabilities",
    "SALÁRIOS E ORDENADOS A PAGAR":                "salaries_payable",
    "PRÓ-LABORE A PAGAR":                          "pro_labore_payable",
    "PARTIC DE LUCROS A PAGAR":                    "profit_sharing_payable",
    "OBRIGAÇÕES SOCIAIS":                          "social_obligations",
    "INSS A RECOLHER":                             "social_security_liabilities",
    "FGTS A RECOLHER":                             "fgts_liabilities",
    "IRRF SOBRE SALÁRIOS":                         "irrf_on_salaries",
    "PROVISÕES":                                   "provisions",
    "PROVISÕES PARA FÉRIAS":                       "provisions_vacation",
    "INSS SOBRE PROVISÕES PARA FÉRIAS":            "inss_on_vacation_provisions",
    "FGTS SOBRE PROVISÕES PARA FÉRIAS":            "fgts_on_vacation_provisions",
    "PIS SOBRE PROVISÕES PARA 13º SALÁRIO":        "pis_on_13th_salary_provisions",
    "OUTRAS OBRIGAÇÕES":                           "other_liabilities",
    "CONTAS A PAGAR":                              "accounts_payable",
    "CARTÃO DE CREDITO SICOOB A PAGAR":            "credit_card_payable_sicoob",
    "TRANSITÓRIA - CARTÃO DE CREDITO SICOOB A PAGAR": "transitory_credit_card_payable_sicoob",
    "PASSIVO NÃO-CIRCULANTE":                      "non_current_liabilities",
    "ECOLÓGICA IND. E COM. DE PROD. SUST.":        "other_non_current_liabilities",
    "LUCROS OU PREJUÍZOS ACUMULADOS":              "retained_earnings",
    "LUCROS ACUMULADOS":                           "retained_earnings",
    "(-) PREJUÍZOS ACUMULADOS":                    "accumulated_losses",
    
    # totais do BP
    "ATIVO":                                       "total_assets",
    "PASSIVO":                                     "total_liabilities",
    "PATRIMÔNIO LÍQUIDO":                          "equity"
}

ACCOUNT_MAP.update({
    # DRE (Income Statement) – JJMAX
    "(-) CANCELAMENTO E DEVOLUÇÕES":              "sales_returns",
    "(-) CUSTOS DE MERCADORIAS ADQUIRIDAS":       "cogs",
    "(-) CUSTOS DAS MERCADORIAS VENDIDAS":        "cogs",
    "(-) DESPESAS COM PESSOAL":                   "personnel_expenses_admin",
    "(-) ALUGUÉIS E ARRENDAMENTOS":               "rent_and_lease_expenses",
    "PREJUÍZO DO EXERCÍCIO":                      "net_income",

    # Balanço (Balance Sheet) – JJMAX
    "ADIANTAMENTOS DE CLIENTES":                  "advances_from_customers",
    "ADIANTAMENTOS A FORNECEDORES":               "advances_to_suppliers",
    "CAPITAL SOCIAL":                             "share_capital",
    "CAPITAL SUBSCRITO":                          "subscribed_capital",
    "(-) CAPITAL A INTEGRALIZAR":                 "capital_not_paid",
    "EMPRÉSTIMOS":                                "loans_payable",
    "EMPRÉSTIMO BANCO ITAU":                      "loans_payable",
    "EMPRÉSTIMOS E FINANCIAMENTOS":               "loans_and_financing",
    "CAIXA":                                      "cash",
    "CAIXA GERAL":                                "cash",
    "BANCO DO BRASIL":                            "bank_accounts",
    "BANCO ITAU UNIBANCO":                        "bank_accounts",
    "BANCO ITAU S/A":                             "bank_accounts",
    "CAIXA ECONÔMICA FEDERAL":                    "bank_accounts",

    # Ativo
    "ATIVO CIRCULANTE DISPONÍVEL":                 "current_assets",
    "PRODUTOS ACABADOS":                           "inventory_finished_goods",
    "APLICAÇÕES FINANCEIRAS LIQUIDEZ IMEDIATA":    "cash_equivalents",
    "IRRF S/ APLICAÇÕES FINANCEIRAS A RECUPERAR":  "taxes_recoverable_financial",

    # Estoques / remessas / consignações
    "REMESSA EM CONSIGNAÇÃO":                      "inventory_on_consignment",
    "SIMPLES REMESSA":                             "simples_remittance",
    "(-) SIMPLES REMESSA":                         "simples_remittance",

    # Tributos a recolher
    "PIS A RECOLHER":                              "tax_liabilities_pis",
    "COFINS A RECOLHER":                           "tax_liabilities_cofins",
    "ICMS A RECOLHER":                             "tax_liabilities_icms",
    "ICMS SUBSTITUIÇÃO TRIBUTÁRIA A RECOLHER":      "tax_liabilities_icms_substitution",
    "IRRF S/ ALUGUEL A RECOLHER":                  "tax_liabilities_irrf_rent",
    "SIMPLES NACIONAL A RECOLHER":                  "tax_liabilities_simples",
    "PARCELAMENTO DE IMPOSTOS FEDERAIS":           "tax_liabilities_federal_installments",
    "PARCELAMENTO DE IMPOSTOS ESTADUAIS":          "tax_liabilities_state_installments",
    "CONTRIBUIÇÃO SINDICAL A RECOLHER":            "tax_liabilities_union",

    # Imobilizado / outros ativos não-correntes
    "CONSORCIOS":                                  "other_non_current_assets",
    "TRANSITORIA DE IMOBILIZADO":                  "other_non_current_assets",
    "CONTAS DE COMPENSAÇÃO ATIVA":                 "memorandum_accounts",

    # Adiantamentos
    "ADIANTAMENTOS A CLIENTES":                    "advances_from_customers",
    "ADIANTAMENTOS DE CLIENTES":                   "advances_from_customers",

    # Despesas diversas
    "ALUGUÉIS A PAGAR":                           "rent_and_lease_expenses",

    # Capital
    "CAPITAL A INTEGRALIZAR":                      "capital_not_paid",

    # Contas de sócios / partes relacionadas
    "CONTA CORRENTE SÓCIOS":                       "shareholder_current_accounts",
    "OUTROS DÉBITOS COM SÓCIOS, ADM, PESSOAS":     "related_party_liabilities",
    "CONTROLADORA, CONTROLADAS E COLIGADAS":       "related_party_receivables",
    "ADVANCED LABS":                               "related_party_receivables",
    "FARMACIA DE MANIPULAÇ GRACIOSA":              "related_party_receivables",
    "FARMACIA MAJESTIC LTDA":                      "related_party_receivables",
    "LAYNESKIN":                                   "related_party_receivables",
    "LAYNESKIN DERMOCOSMESTICOS LTDA":             "related_party_receivables",
    "MERCADO OFICIAL":                             "related_party_receivables",
    "OFICIAL MF ADMINISTRAÇ E PARTIC":             "related_party_receivables",
    "OFH INVESTIMENTOS E PARTICIPAÇ":              "related_party_receivables",
    "ELAINE VERISSIMO":                            "related_party_liabilities",

    "IRRF S/ FOLHA A RECOLHER":              "tax_liabilities_irrf",
    "ENERGIA ELÉTRICA, ÁGUA E TELEFONE A PAGA": "utilities_payable",

    # DRE – despesas novas
    "(-) COMISSÕES SOBRE VENDAS":    "commission_expenses",
    "(-) SERVIÇOS TOMADOS DE PJ":    "services_from_third_parties",

    # BP – contas a receber / adiantamentos
    "ADVANCED LABS LTDA":            "related_party_receivables",
    "FARMACIA MAJESTIC":             "related_party_receivables",
    "OFC COMERCIO - LOG":            "related_party_receivables",
    "CLEVERSON SANTOS LIMA":         "related_party_receivables",
    "DROGARIA E FARM DE MANIP VILA AMERICA": "related_party_receivables",

    "ADIANTAMENTO DE SALÁRIO":       "advances_to_employees",
    "ADIANTAMENTO A EMPREGADOS":     "advances_to_employees",
    "EMPRÉSTIMO A EMPREGADOS":       "loans_to_employees",
    "EMPRÉSTIMOS A FUNCIONÁRIOS":    "loans_to_employees",

    "ADIANTAMENTO DE CLIENTES - MERCADO OFICIAL": "advances_from_customers",

    # Tributos
    "IRRF S/ APLICAÇÃO FINANCEIRA A RECUPERAR":   "taxes_recoverable_financial",
    "ISS A RECOLHER":                             "tax_liabilities_iss",
    "SUBSTITUIÇÃO TRIBUTÁRIA A RECOLHER":         "tax_liabilities_icms_substitution",
    "ICMS ANTECIPADO A RECOLHER":                 "tax_liabilities_icms",

    # Imobilizado / outros ativos
    "INSTALAÇÕES":                                "fixed_assets_other",
    "ATIVO PERMANENTE":                           "non_current_assets",

    # Sócios / contas correntes
    "CONTA CORRENTE DE SOCIOS":                   "shareholder_current_accounts",

    # Benefícios em terceiros?
    "BENF. EM MOVEIS DE TERC":                    "other_operating_expenses",

    "FARMACIA DE MANIP GRACIOSA":                 "related_party_receivables"
})

@st.cache_data
def load_csv_from_dropbox(filename: str, expected_cols: list[str]) -> pd.DataFrame | None:
    path = f"{BASE_PATH}/{filename}"
    try:
        _, res = dbx.files_download(path=path)
    except dropbox.exceptions.ApiError as e:
        st.warning(f"Arquivo não encontrado: {filename}")
        return None
    df = pd.read_csv(BytesIO(res.content))
    if missing := set(expected_cols) - set(df.columns):
        st.error(f"Colunas faltando em {filename}: {missing}")
        return None
    return df

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COMMON_COLUMNS).copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df

def add_metadata(df, stmt, ref_date, cid):
    df["statement"]  = stmt
    df["ref_date"]   = pd.to_datetime(ref_date)
    df["company_id"] = cid
    return df

def apply_account_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df["account_std"] = df["account"].map(ACCOUNT_MAP)
    unmapped = set(df["account"]) - set(ACCOUNT_MAP.keys())
    if unmapped:
        st.warning(f"⚠️ Contas não mapeadas: {unmapped}")
    df["account_std"] = df["account_std"].fillna("outros")
    return df

def clean_data(df):
    return df.dropna(subset=["amount"]).query("amount != 0").drop_duplicates()

def load_and_clean(company_id: str, date_str: str) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    dre_raw = load_csv_from_dropbox(f"DRE_{date_str}_{company_id}.csv",
                                    ["nome_empresa","descrição","valor"])
    bal_raw = load_csv_from_dropbox(f"BALANCO_{date_str}_{company_id}.csv",
                                    ["nome_empresa","descrição","saldo_atual"])
    if dre_raw is None or bal_raw is None:
        return None, None

    dre = clean_data(
        apply_account_mapping(
            add_metadata(standardize_columns(dre_raw), "income_statement", date_str, company_id)
        )
    )
    bal = clean_data(
        apply_account_mapping(
            add_metadata(standardize_columns(bal_raw), "balance_sheet",    date_str, company_id)
        )
    )
    return dre, bal

# -----------------------------------------------------------------------------
# 3. Cálculo de Indicadores
# -----------------------------------------------------------------------------
def compute_indicators(dre: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    dre_sum       = dre.groupby("account_std")["amount"].sum()
    receita_liq   = dre_sum.get("net_revenue", 0)
    custo_vendas  = dre_sum.get("cogs", 0)
    lucro_bruto   = dre_sum.get("gross_profit", receita_liq - custo_vendas)
    despesas_op   = dre_sum.get("operating_expenses", 0)
    ebitda        = lucro_bruto - despesas_op
    lucro_liq     = dre_sum.get("net_income", 0)

    bal_sum        = bal.groupby("account_std")["amount"].sum()
    ativo_circ     = bal_sum.get("current_assets", 0)
    estoque        = bal_sum.get("inventory", 0)
    pass_circ      = bal_sum.get("current_liabilities", 0)
    non_current    = bal_sum.get("non_current_assets", 0)
    total_ativo    = bal_sum.get("total_assets", ativo_circ + non_current)
    non_current_li = bal_sum.get("non_current_liabilities", 0)
    pat_liq        = bal_sum.get("equity", 0)
    total_pass     = pass_circ + non_current_li

    liquidez_corrente = ativo_circ / pass_circ if pass_circ else None
    liquidez_seca     = (ativo_circ - estoque) / pass_circ if pass_circ else None
    endividamento     = total_pass / total_ativo if total_ativo else None
    roa               = lucro_liq / total_ativo if total_ativo else None
    roe               = lucro_liq / pat_liq if pat_liq else None

    return pd.DataFrame({
        "Indicador": [
            "Lucro Bruto", "EBITDA", "Lucro Líquido",
            "Liquidez Corrente", "Liquidez Seca", "Endividamento",
            "ROA", "ROE"
        ],
        "Valor": [
            lucro_bruto, ebitda, lucro_liq,
            liquidez_corrente, liquidez_seca, endividamento,
            roa, roe
        ]
    })

# -----------------------------------------------------------------------------  
# 4. UI & Navigation  
# -----------------------------------------------------------------------------
if st.session_state.get("logged_in"):
    username = st.session_state["username"]
    user_info = USERS[username]
    role      = user_info["role"]
    empresa   = user_info["empresa"]

    authenticator.logout("Sair", "sidebar")
    st.sidebar.success(f"Conectado como {user_info['name']} ({role})")

    available_companies = ["CICLOMADE", "JJMAX", "SAUDEFORMA"]
    if role == "admin":
        session_companies = st.sidebar.multiselect(
            "Selecione empresas", available_companies, default=available_companies
        )
    else:
        session_companies = [empresa]

    # Filtrar apenas empresas válidas
    session_companies = [c for c in session_companies if c in available_companies]
    if not session_companies:
        st.sidebar.error("Selecione ao menos uma empresa válida.")

    session_date = st.sidebar.date_input("Data de Referência", value=pd.to_datetime("2024-12-31"))
    date_str = session_date.strftime("%Y-%m-%d")

    company_for_metrics = st.sidebar.selectbox("Empresa para Métricas", session_companies)

    # Carregar dados com tratamento de erros
    all_dre, all_bal = [], []
    for comp in session_companies:
        dre, bal = load_and_clean(comp, date_str)
        if dre is None or bal is None:
            st.warning(f"Pulando {comp}: dados não encontrados.")
            continue
        all_dre.append(dre)
        all_bal.append(bal)

    if all_dre and all_bal:
        df_all = pd.concat(all_dre + all_bal, ignore_index=True)
    else:
        df_all = pd.DataFrame()  # vazio

    page = st.sidebar.radio("📊 Navegação", ["Visão Geral", "Dashboards", "Chatbot"])

    if page == "Visão Geral":
        dre_sel, bal_sel = load_and_clean(company_for_metrics, date_str)
        if dre_sel is None or bal_sel is None:
            st.error("Não há dados para Visão Geral.")
        else:
            rpt = compute_indicators(dre_sel, bal_sel)
            st.header(f"🏁 Indicadores {company_for_metrics} em {date_str}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Lucro Bruto",  f"R$ {rpt.loc[0,'Valor']:,.2f}")
            c2.metric("EBITDA",        f"R$ {rpt.loc[1,'Valor']:,.2f}")
            c3.metric("Lucro Líquido", f"R$ {rpt.loc[2,'Valor']:,.2f}")
            c4, c5, c6 = st.columns(3)
            c4.metric("Liquidez Corrente", f"{rpt.loc[3,'Valor']:.2f}")
            c5.metric("Endividamento",     f"{rpt.loc[5,'Valor']:.2%}")
            c6.metric("ROE",               f"{rpt.loc[7,'Valor']:.2%}")

            st.markdown("---")
            st.dataframe(rpt.style.format({"Valor":"R$ {:,.2f}"}), use_container_width=True)

    elif page == "Dashboards":
        if df_all.empty:
            st.info("Nenhum dado disponível para as seleções atuais.")
        else:
            st.header("📈 Dashboards")
            # Exemplo de gráfico Altair
            chart = alt.Chart(df_all).mark_bar().encode(
                x="account_std:N",
                y="amount:Q",
                color="company_id:N",
                tooltip=["company_id","account_std","amount"]
            )
            st.altair_chart(chart, use_container_width=True)

    else:  # Chatbot
        st.markdown(
            """
            <style>
            .stApp { background-color: #191919; font-family: 'Segoe UI', sans-serif; }
            .stChatMessage.user { background-color: #d1e7ff; border-radius: 12px; padding: 10px; color: #003366; }
            .stChatMessage.assistant { background-color: #ffffff; border-radius: 12px; padding: 10px; border: 1px solid #e0e0e0; color: #222; }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.header(f"🤖 Chatbot Contábil - {company_for_metrics}")

        # Inicializa histórico
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Renderiza histórico rolável
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar=msg.get("avatar", None)):
                st.markdown(msg["content"])

        # Entrada do usuário
        if prompt := st.chat_input("Digite sua pergunta sobre os indicadores..."):
            # Adiciona pergunta do usuário
            st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🧑"})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)

            # Busca contexto semântico
            contexts = semantic_search(prompt, index, meta, top_k=3)
            ctx_txt = "\n".join(f"Q: {c['q']}\nA: {c['a']}" for c in contexts)

            # Carrega dados brutos
            dre_raw = load_csv_from_dropbox(
                f"DRE_{date_str}_{company_for_metrics}.csv",
                ["nome_empresa", "descrição", "valor"]
            )
            bal_raw = load_csv_from_dropbox(
                f"BALANCO_{date_str}_{company_for_metrics}.csv",
                ["nome_empresa", "descrição", "saldo_atual"]
            )
            if dre_raw is None or bal_raw is None:
                st.error("Não foi possível carregar os dados brutos.")
                st.stop()

            dre_csv = dre_raw.to_csv(index=False)
            bal_csv = bal_raw.to_csv(index=False)

            # Monta prompt
            full_prompt = f"""
Você é um assistente contábil.

Aqui estão os dados brutos da Demonstração de Resultados (DRE):
{dre_csv}

E aqui os dados brutos do Balanço Patrimonial:
{bal_csv}

Contextos anteriores (semânticos):
{ctx_txt}

Pergunta: {prompt}

Responda de forma objetiva e fundamentada **nos dados brutos acima**.
"""

            # Efeito de digitação
            with st.chat_message("assistant", avatar="🤖"):
                typing_placeholder = st.empty()
                typing_placeholder.markdown("_Digitando..._")
                time.sleep(0.8)  # pequena pausa para simular início
                resposta = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Assistente contábil de indicadores."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0
                ).choices[0].message.content.strip()

                # Simula texto aparecendo aos poucos
                typing_placeholder.empty()
                displayed_text = ""
                for char in resposta:
                    displayed_text += char
                    typing_placeholder.markdown(displayed_text)
                    time.sleep(0.005)  # velocidade da digitação

            # Salva no histórico
            st.session_state.messages.append({"role": "assistant", "content": resposta, "avatar": "🤖"})

            # Resposta encadeada: sugestão de próxima pergunta
            follow_up = f"Quer que eu analise também a evolução desses indicadores em relação ao período anterior?"
            st.session_state.messages.append({"role": "assistant", "content": follow_up, "avatar": "🤖"})
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(follow_up)

            # Armazena embedding
            upsert_embedding(prompt, resposta, index, meta)