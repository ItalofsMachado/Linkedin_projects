"""
AML Detection Pipeline - Detecção de Lavagem de Dinheiro
========================================================

Pipeline híbrido para detecção de fraude em transações bancárias utilizando:
- Regras de negócio
- Isolation Forest (análise estatística)
- Graph-based scoring (propagação de risco)

Dataset: HI-Small_Trans.csv

Autor: Italo Machado
Data: Abril 2026
"""

import polars as pl
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import gc
import os
from datetime import datetime

# ========================= CONFIGURAÇÕES =========================
DATA_PATH = "HI-Small_Trans.csv"

# Pesos da fusão final (ajustados para dar mais importância ao Isolation Forest)
W_REGRAS = 0.15
W_IFOREST = 0.50
W_GRAFO = 0.35

# Parâmetros do Grafo
MAX_CAMADAS = 3
ALPHA = 0.55

# Configurações de Threshold (quanto menor o percentil, mais alertas)
THRESHOLD_REGRAS = 0.995
THRESHOLD_IFOREST = 0.98
THRESHOLD_GRAFO = 0.98
THRESHOLD_HIBRIDO = 0.97

print("🚀 Iniciando Pipeline de Detecção de Lavagem de Dinheiro")
print(f"   Horário: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ========================= CARREGAMENTO DE DADOS =========================
print("📂 Carregando dados...")
df = pl.read_csv(DATA_PATH, try_parse_dates=True)

if "Account_duplicated_0" in df.columns:
    df = df.rename({"Account_duplicated_0": "Account_dest"})

# Otimização de memória
df = df.with_columns([pl.col("Amount Received").cast(pl.Float32)])

print(f"✅ Dados carregados: {df.height:,} transações | {df.select('Account').n_unique():,} contas únicas")


# ========================= TARGET (Ground Truth) =========================
df_target = df.group_by("Account").agg([
    pl.col("Is Laundering").max().alias("target")
])


# ========================= 1. REGRAS DE NEGÓCIO =========================
print("📏 Aplicando regras de negócio...")

df = df.with_columns([
    (pl.col("Account") == pl.col("Account_dest")).alias("self_loop_bool")
])

# Cálculo de tempo entre transações
df_time = (
    df.filter(pl.col("self_loop_bool") == False)
    .sort(["Account", "Timestamp"])
    .with_columns([
        (pl.col("Timestamp").diff().over("Account").dt.total_seconds() / 60)
        .fill_null(9999)
        .alias("time_diff_min")
    ])
)

df = df.join(
    df_time.select(["Timestamp", "Account", "time_diff_min"]),
    on=["Timestamp", "Account"],
    how="left"
).fill_null(9999)

df = df.with_columns([
    (pl.col("time_diff_min") < 5).cast(pl.Int8).alias("burst_flag")
])

# Agregação por conta
df_conta = df.group_by("Account").agg([
    pl.len().alias("qtd_operacoes"),
    pl.col("Account_dest").n_unique().alias("qtd_contrapartes"),
    pl.col("burst_flag").sum().alias("qtd_burst"),
    pl.col("self_loop_bool").mean().alias("self_loop_ratio"),
])

df_conta = df_conta.with_columns([
    (pl.col("qtd_contrapartes") / pl.col("qtd_operacoes")).alias("ratio_dispersion")
])

df_conta = df_conta.with_columns([
    (pl.col("qtd_burst") / 5).clip(0, 1).alias("score_burst"),
    (pl.col("self_loop_ratio") / 0.3).clip(0, 1).alias("score_self_loop"),
    pl.col("ratio_dispersion").clip(0, 1).alias("score_dispersion")
])

df_conta = df_conta.with_columns([
    (pl.col("score_burst") * 0.3 +
     pl.col("score_self_loop") * 0.3 +
     pl.col("score_dispersion") * 0.4).alias("score_regras_norm")
])

print("✅ Regras de negócio concluídas")


# ========================= 2. ISOLATION FOREST =========================
print("🌲 Treinando Isolation Forest...")

df = df.with_columns([
    (pl.col("Amount Received") / (pl.col("Amount Received").mean().over("Account") + 1e-6)).alias("rel_amount"),
    pl.col("Account_dest").n_unique().over("Account").alias("bank_dispersion"),
    ((pl.col("Amount Received") - pl.col("Amount Received").mean().over("Account")) /
     (pl.col("Amount Received").std().over("Account") + 1e-6)).alias("val_zscore")
])

# Treinamento com amostra para maior velocidade
df_sample = df.sample(fraction=0.3, seed=42)
X_train = df_sample.select(["rel_amount", "bank_dispersion", "val_zscore", "Amount Received"]).fill_null(0).to_numpy()

iso = IsolationForest(n_estimators=150, contamination=0.007, random_state=42, n_jobs=-1)
iso.fit(X_train)

del X_train, df_sample
gc.collect()

# Scoring na base completa
X_full = df.select(["rel_amount", "bank_dispersion", "val_zscore", "Amount Received"]).fill_null(0).to_numpy()
scores = iso.decision_function(X_full)

df = df.with_columns([pl.Series("score_iforest_raw", scores)])
df = df.with_columns([(-pl.col("score_iforest_raw")).alias("score_iforest_risk")])

del X_full, scores
gc.collect()

# Normalização
min_s = df.select(pl.col("score_iforest_risk").min()).item()
max_s = df.select(pl.col("score_iforest_risk").max()).item()

df = df.with_columns([
    ((pl.col("score_iforest_risk") - min_s) / (max_s - min_s + 1e-8))
    .alias("score_iforest_norm")
])

df_iforest = df.group_by("Account").agg([
    pl.col("score_iforest_norm").mean().alias("mean_iforest"),
    pl.col("score_iforest_norm").max().alias("max_iforest")
]).with_columns([
    (pl.col("max_iforest") * 0.7 + pl.col("mean_iforest") * 0.3)
    .alias("score_iforest_final")
])

print("✅ Isolation Forest concluído")


# ========================= 3. GRAFO =========================
print("🕸️ Construindo grafo...")

df_edges = df.select(["Account", "Account_dest", "Amount Received"]).sample(fraction=0.35, seed=42).to_pandas()

G = nx.DiGraph()
edges_grouped = df_edges.groupby(['Account', 'Account_dest'], as_index=False)['Amount Received'].sum()
for _, row in edges_grouped.iterrows():
    G.add_edge(row['Account'], row['Account_dest'], weight=row['Amount Received'])

print(f"   → Grafo criado: {len(G.nodes()):,} nós | {len(G.edges()):,} arestas")


def score_grafo(G, max_camadas=3, alpha=0.55):
    """Calcula score baseado em propagação de risco no grafo."""
    scores = {}
    nodes_list = list(G.nodes())[:70000]  # Limite para performance
    
    for node in nodes_list:
        visitados = {node}
        fronteira = {node}
        score = 0.0
        peso_total = 0.0
        
        for camada in range(1, max_camadas + 1):
            prox = set()
            for n in fronteira:
                prox.update(G.successors(n))
                prox.update(G.predecessors(n))
            prox -= visitados
            if not prox:
                break
            peso = alpha ** (camada - 1)
            score += len(prox) * peso
            peso_total += peso
            visitados |= prox
            fronteira = prox
        
        scores[node] = score / peso_total if peso_total > 0 else 0.0
    
    df_g = pd.DataFrame(scores.items(), columns=["Account", "score_grafo"])
    return df_g


df_grafo = score_grafo(G, MAX_CAMADAS, ALPHA)
df_grafo["score_grafo_norm"] = np.sqrt(df_grafo["score_grafo"])  # Normalização suave
df_grafo = pl.from_pandas(df_grafo)

print("✅ Cálculo do grafo concluído")


# ========================= 4. FUSÃO DOS SCORES =========================
print("🔗 Realizando fusão dos modelos...")

df_final = (
    df_conta
    .join(df_iforest, on="Account")
    .join(df_grafo, on="Account", how="left")
    .join(df_target, on="Account")
)

df_final = df_final.fill_null(0)

df_final = df_final.with_columns([
    (
        pl.col("score_regras_norm") * W_REGRAS +
        pl.col("score_iforest_final") * W_IFOREST +
        pl.col("score_grafo_norm") * W_GRAFO
    ).alias("score_final")
])


# ========================= 5. THRESHOLDS E FLAGS =========================
def aplicar_threshold(df, col_score, nome_flag, perc=0.98):
    """Aplica threshold baseado em percentil."""
    t = df.select(pl.col(col_score).quantile(perc)).item()
    print(f"   Threshold {nome_flag}: {t:.4f} (percentil {perc})")
    return df.with_columns([(pl.col(col_score) >= t).cast(pl.Int8).alias(nome_flag)])


print("📊 Aplicando thresholds...")
df_final = aplicar_threshold(df_final, "score_regras_norm",   "flag_regras",   perc=THRESHOLD_REGRAS)
df_final = aplicar_threshold(df_final, "score_iforest_final", "flag_iforest", perc=THRESHOLD_IFOREST)
df_final = aplicar_threshold(df_final, "score_grafo_norm",    "flag_grafo",   perc=THRESHOLD_GRAFO)
df_final = aplicar_threshold(df_final, "score_final",         "flag_hibrido", perc=THRESHOLD_HIBRIDO)


# ========================= RELATÓRIO FINAL =========================
resumo = df_final.group_by("target").agg([
    pl.len().alias("total_contas"),
    pl.col("flag_regras").sum().alias("detect_regras"),
    pl.col("flag_iforest").sum().alias("detect_iforest"),
    pl.col("flag_grafo").sum().alias("detect_grafo"),
    pl.col("flag_hibrido").sum().alias("detect_hibrido"),
    (pl.col("flag_hibrido").mean() * 100).round(2).alias("taxa_hibrido_%")
]).with_columns([
    pl.when(pl.col("target") == 1).then(pl.lit("Com Fraude")).otherwise(pl.lit("Sem Fraude")).alias("Grupo")
])

print("\n" + "="*85)
print("RESUMO COMPARATIVO FINAL")
print("="*85)
print(resumo.to_pandas())


# ========================= CONSENSO =========================
df_final = df_final.with_columns([
    (pl.col("flag_regras") + pl.col("flag_iforest") + pl.col("flag_grafo"))
    .alias("qtd_modelos")
])


# ========================= EXPORTAÇÃO =========================
print("\n💾 Exportando arquivos...")

resumo.write_excel("resumo_comparativo_fraude.xlsx")
df_final.filter(pl.col("flag_hibrido") == 1).write_excel("contas_suspeitas_hibrido.xlsx")

print("✅ Arquivos Excel exportados com sucesso!")


# ========================= GRÁFICOS =========================
print("📊 Gerando gráficos...")

plt.style.use("ggplot")

df_plot = df_final.group_by("target").agg([
    pl.col("flag_regras").mean().alias("Regras"),
    pl.col("flag_iforest").mean().alias("Isolation Forest"),
    pl.col("flag_grafo").mean().alias("Grafo"),
    pl.col("flag_hibrido").mean().alias("Híbrido")
]).to_pandas()

df_consenso = df_final.group_by("qtd_modelos").agg(
    pl.len().alias("quantidade")
).to_pandas()

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

df_plot.set_index("target").plot(kind='bar', ax=axs[0])
axs[0].set_title("Taxa de Detecção - Com Fraude vs Sem Fraude")
axs[0].set_ylabel("Taxa de Detecção")
axs[0].tick_params(axis='x', rotation=0)

axs[1].bar(df_consenso["qtd_modelos"], df_consenso["quantidade"], color='skyblue', edgecolor='black')
axs[1].set_title("Consenso entre os Modelos")
axs[1].set_xlabel("Quantidade de Modelos que Detectaram")
axs[1].set_ylabel("Número de Contas")

plt.tight_layout()
plt.savefig("graficos_analise_fraude.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 Pipeline finalizado com sucesso!")
print("📁 Arquivos gerados:")
print("   • resumo_comparativo_fraude.xlsx")
print("   • contas_suspeitas_hibrido.xlsx")
print("   • graficos_analise_fraude.png")