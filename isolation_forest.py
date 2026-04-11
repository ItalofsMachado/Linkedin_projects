"""
🚀 Detecção de Fraude com Isolation Forest + Regras de Negócio + Explicabilidade SHAP
===============================================================================

Este script detecta transações suspeitas utilizando:
- Regras de negócio (Burst e Self-Loop)
- Modelo de Machine Learning (Isolation Forest)
- Análise de "Pontos Cegos" com SHAP

Autor: Italo Machado
linkedin : https://www.linkedin.com/in/italo-machado/
Data: 10/04/2026
"""

import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
from sklearn.ensemble import IsolationForest

# ========================= CONFIGURAÇÕES =========================

# Caminhos
DATA_PATH = "HI-Small_Trans.csv"
OUTPUT_DIR = "explicabilidade_fraude"

# Features para o modelo
FEATURES_IA = ["rel_amount", "bank_dispersion", "val_zscore", "Amount Received"]

# Parâmetros do modelo
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "contamination": 0.002,
    "random_state": 42,
    "n_jobs": -1
}


# ========================= FUNÇÕES AUXILIARES =========================

def log(mensagem: str) -> None:
    """Função de log com timestamp e emoji para melhor experiência."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] 🚀 {mensagem}")


def criar_diretorio_saida() -> None:
    """Cria o diretório de saída se não existir."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        log(f"Diretório de saída criado: {OUTPUT_DIR}")
    else:
        log(f"Diretório de saída: {OUTPUT_DIR}")


# ========================= CARREGAMENTO E PRÉ-PROCESSAMENTO =========================

def carregar_dados() -> pl.DataFrame:
    """Carrega os dados usando Polars (muito mais rápido que pandas)."""
    log("Carregando base de dados com Polars...")
    df = pl.read_csv(DATA_PATH, try_parse_dates=True)

    # Renomeia coluna se necessário
    if "Account_duplicated_0" in df.columns:
        df = df.rename({"Account_duplicated_0": "Account_dest"})
        log("Coluna 'Account_duplicated_0' renomeada para 'Account_dest'.")

    return df


def aplicar_regras_negocio(df: pl.DataFrame) -> pl.DataFrame:
    """Aplica regras de negócio: Self-Loop e Burst (transações rápidas)."""
    log("Aplicando regras de negócio: Self-Loop e Burst...")

    # Self-Loop
    df = df.with_columns([
        (pl.col("Account") == pl.col("Account_dest")).alias("self_loop_bool")
    ])

    # Calcula diferença de tempo entre transações (apenas não self-loop)
    df_time = (
        df.filter(pl.col("self_loop_bool") == False)
        .sort(["Account", "Timestamp"])
        .with_columns([
            (pl.col("Timestamp").diff().over("Account").dt.total_seconds() / 60)
            .fill_null(9999)
            .alias("time_diff_min")
        ])
    )

    # Junta de volta e cria flag de burst
    df = df.join(
        df_time.select(["Timestamp", "Account", "time_diff_min"]),
        on=["Timestamp", "Account"],
        how="left"
    ).fill_null(9999)

    df = df.with_columns([
        (pl.col("time_diff_min") < 5).cast(pl.Int8).alias("burst_flag")
    ])

    return df


# ========================= FEATURE ENGINEERING =========================

def engenharia_features(df: pl.DataFrame) -> pl.DataFrame:
    """Cria features enriquecidas para o modelo de IA."""
    log("Realizando Feature Engineering...")

    df = df.with_columns([
        # Valor relativo à média da conta
        (pl.col("Amount Received") / 
         (pl.col("Amount Received").mean().over("Account") + 1e-6))
        .alias("rel_amount"),

        # Dispersão de bancos usados
        pl.col("To Bank").n_unique().over("Account").alias("bank_dispersion"),

        # Z-Score do valor recebido
        ((pl.col("Amount Received") - pl.col("Amount Received").mean().over("Account")) /
         (pl.col("Amount Received").std().over("Account") + 1e-6))
        .alias("val_zscore")
    ])

    return df


# ========================= TREINAMENTO DO MODELO =========================

def treinar_modelo(df: pl.DataFrame) -> tuple[IsolationForest, pl.DataFrame]:
    """Treina o Isolation Forest e adiciona scores e predições ao DataFrame."""
    log(f"Treinando Isolation Forest em {len(df):,} registros...")

    # Prepara matriz de features
    X = (
        df.select(FEATURES_IA)
        .fill_null(0)
        .cast(pl.Float32)
        .to_numpy()
    )

    inicio = time.time()
    iso = IsolationForest(**ISOLATION_FOREST_PARAMS)
    iso.fit(X)

    tempo_treino = time.time() - inicio
    log(f"Modelo treinado com sucesso em {tempo_treino:.2f} segundos.")

    # Predições e scores
    scores = iso.decision_function(X)
    preds = iso.predict(X)

    df = df.with_columns([
        pl.Series("score_ia", scores),
        pl.Series("is_anomaly_ia", preds)
    ])

    return iso, df


# ========================= EXPLICABILIDADE (SHAP) =========================

def analisar_pontos_cegos(df: pl.DataFrame, modelo: IsolationForest) -> None:
    """Identifica e explica 'Pontos Cegos' (detectados pela IA, mas não pelas regras)."""
    log("🔍 Iniciando análise de Pontos Cegos com SHAP...")

    # Agrupa por conta para identificar pontos cegos
    df_contas = df.group_by("Account").agg([
        pl.col("burst_flag").sum().alias("qtd_burst"),
        pl.col("self_loop_bool").sum().alias("qtd_self_loop"),
        pl.col("is_anomaly_ia").min().alias("flag_ia")
    ]).filter(
        (pl.col("qtd_burst") == 0) &
        (pl.col("qtd_self_loop") == 0) &
        (pl.col("flag_ia") == -1)
    )

    lista_contas_ponto_cego = df_contas.select("Account").to_series().to_list()

    if not lista_contas_ponto_cego:
        log("⚠️ Nenhuma conta classificada como Ponto Cego.")
        return

    df_ponto_cego = df.filter(pl.col("Account").is_in(lista_contas_ponto_cego))
    log(f"Encontradas {len(df_ponto_cego):,} transações em Pontos Cegos.")

    X_ponto_cego = (
        df_ponto_cego.select(FEATURES_IA)
        .fill_null(0)
        .cast(pl.Float32)
        .to_numpy()
    )

    explainer = shap.TreeExplainer(modelo)

    # === Gráfico Global ===
    n_sample = min(1000, len(X_ponto_cego))
    X_sample = X_ponto_cego[:n_sample]
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURES_IA, show=False)
    plt.title("Ponto Cego: Por que o modelo detectou o que as regras não viram?", 
              fontsize=15, pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_shap_ponto_cego_global.png", dpi=300, bbox_inches='tight')
    plt.close()
    log("✅ Gráfico SHAP Global dos Pontos Cegos salvo.")

    # === Waterfall das Top 3 contas mais anômalas ===
    df_top = df_ponto_cego.sort("score_ia").unique(subset=["Account"]).limit(3)

    for i, registro in enumerate(df_top.to_dicts()):
        X_exemplo = np.array([[registro[f] for f in FEATURES_IA]])

        shap_val = explainer.shap_values(X_exemplo)

        plt.figure(figsize=(12, 5))
        exp = shap.Explanation(
            values=shap_val[0],
            base_values=explainer.expected_value,
            data=X_exemplo[0],
            feature_names=FEATURES_IA
        )

        shap.plots.waterfall(exp, show=False)
        conta_id = str(registro['Account'])[:12]
        plt.title(f"🔍 Ponto Cego #{i+1} - Conta: {conta_id}...", 
                  fontsize=14, loc='left', pad=25)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/justificativa_ponto_cego_{i+1}.png", dpi=300, bbox_inches='tight')
        plt.close()
        log(f"   → Waterfall da conta {i+1}/3 gerado.")


# ========================= EXECUÇÃO PRINCIPAL =========================

def main() -> None:
    """Função principal que orquestra todo o pipeline."""
    log("🚀 Iniciando pipeline de Detecção de Fraude + Explicabilidade")

    criar_diretorio_saida()

    # 1. Carregamento
    df = carregar_dados()

    # 2. Regras de negócio
    df = aplicar_regras_negocio(df)

    # 3. Feature Engineering
    df = engenharia_features(df)

    # 4. Treinamento
    modelo, df = treinar_modelo(df)

    # 5. Análise de Pontos Cegos com SHAP
    analisar_pontos_cegos(df, modelo)

    # TODO: Inserir aqui sua lógica de exportação para Excel
    log("📊 [TODO] Gerar dashboard final em Excel...")

    log("✨ PIPELINE CONCLUÍDO COM SUCESSO! ✨")
    log(f"Resultados disponíveis em: ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()