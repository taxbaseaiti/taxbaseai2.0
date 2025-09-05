import pandas as pd
import pytest

from app import (
    standardize_columns,
    apply_account_mapping,
    clean_data,
)

# 1. Dados de exemplo
raw = pd.DataFrame({
    "nome_empresa": ["A", "B", None],
    "descrição":      ["X", "Y", "Z"],
    "valor":         ["100", "bad", "300"],
})

# 2. Teste standardize_columns
def test_standardize_columns_converts_and_renames():
    df = standardize_columns(raw)
    # coluna renomeada
    assert "company" in df.columns
    assert "account" in df.columns
    assert "amount"  in df.columns
    # conversão numérica: "bad" vira NaN
    assert df["amount"].dtype == float
    assert pd.isna(df.loc[df["company"]=="B","amount"].iloc[0])

# 3. Teste apply_account_mapping
def test_apply_account_mapping_unmapped_goes_to_outros():
    df = pd.DataFrame({"account": ["RECEITA LÍQUIDA","UNKNOW"]})
    df = apply_account_mapping(df)
    assert df.loc[0,"account_std"] == "receita_liquida"
    assert df.loc[1,"account_std"] == "outros"

# 4. Teste clean_data
def test_clean_data_drops_na_zero_and_duplicates():
    df = pd.DataFrame({
        "amount": [0, 10, None, 10],
        "foo":    [1,   2,   3,     2],
    })
    out = clean_data(df)
    # mantém apenas o 10 único
    assert len(out) == 1
    assert out["amount"].iloc[0] == 10

# 5. Parametrized exemplo de cálculo simples
@pytest.mark.parametrize(
    "revenues, costs, expected_gross",
    [
        ([100,200], [50,  50], 200),
        ([  0],     [   ],    0),
        ([300],    [100,150], 50),
    ]
)
def test_lucro_bruto_calculation(revenues, costs, expected_gross):
    df_rev = pd.DataFrame({"account_std": ["receita_liquida"]*len(revenues),
                           "amount": revenues})
    df_cost= pd.DataFrame({"account_std": ["custo_vendas"]*len(costs),
                           "amount": costs})
    dre_sum = pd.concat([df_rev, df_cost]).groupby("account_std")["amount"].sum()
    receita_liq = dre_sum.get("receita_liquida",0)
    custo_vend  = dre_sum.get("custo_vendas",0)
    assert receita_liq - custo_vend == expected_gross