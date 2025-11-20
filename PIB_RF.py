import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.api import VAR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import RFECV 
from typing import Dict, Any, Tuple, List 
from tqdm import tqdm # Adicionado para a barra de progresso no Bootstrap


# --- Configurações Iniciais ---
warnings.filterwarnings("ignore") 
plt.style.use('seaborn-v0_8-darkgrid') 
TARGET_VAR = 'IBC_Br' # Variável Alvo (Proxy do PIB)
N_TEST_OBS = 12 # Período de teste em meses (1 ano)


# ==============================================================================
# 1. FUNÇÕES DE BUSCA E PRÉ-PROCESSAMENTO DE DADOS
# ==============================================================================

def get_bcb_data(code: int, name: str) -> pd.DataFrame | None:
    """Busca dados de séries temporais no Banco Central do Brasil (BCB) via API."""
    url = f'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json'
    try:
        df = pd.read_json(url)
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df.set_index('data', inplace=True)
        df = df.rename(columns={'valor': name})
        return df.astype(float)
    except Exception as e:
        print(f"❌ Erro ao buscar série {code} ({name}): {e}")
        return None

def prepare_time_series_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Busca, consolida e pré-processa as séries temporais do BCB."""
    print("\n--- 1. 🔍 BUSCA E PRÉ-PROCESSAMENTO DE DADOS ---")
    
    bcb_codes: Dict[int, str] = {
        24363: 'IBC_Br', 
        433: 'IPCA', 
        4390: 'Selic', 
        3698: 'Dolar' 
    }

    # Usando tqdm para mostrar o progresso da busca de dados
    series_list: List[pd.DataFrame] = [
        get_bcb_data(code, name) 
        for code, name in tqdm(bcb_codes.items(), desc="Buscando séries do BCB")
    ]
    df_full = pd.concat([s for s in series_list if s is not None], axis=1).dropna()
    
    df_full = df_full[df_full.index >= '2003-01-01']
    df_diff = df_full.pct_change().dropna() * 100
    df_diff = df_diff.replace([np.inf, -np.inf], np.nan).dropna()
    
    df_treino = df_diff.iloc[:-N_TEST_OBS]
    df_teste = df_diff.iloc[-N_TEST_OBS:]
    
    print(f"✅ Dados Carregados. Treino: {len(df_treino)} meses | Teste: {len(df_teste)} meses")
    
    return df_treino, df_teste


# ==============================================================================
# 2. FEATURE ENGINEERING: CRIAÇÃO DE LAGS (Defasagens)
# ==============================================================================

def find_optimal_lag(df_treino: pd.DataFrame) -> int:
    """Encontra o número de lags ideal usando o critério AIC (VAR)."""
    print("\n--- 2. 🧠 OTIMIZAÇÃO DO LAG (Via VAR e AIC) ---")
    
    modelo_var = VAR(df_treino)
    
    try:
        var_result = modelo_var.fit(maxlags=12, ic='aic', verbose=False)
    except Exception:
        var_result = modelo_var.fit(maxlags=6, ic='aic', verbose=False)
        
    lags_ml = var_result.k_ar
    print(f"✅ Lags Otimizados (k_ar) para Feature Engineering: {lags_ml}")
    return lags_ml

def create_lag_features(df_diff: pd.DataFrame, lags_ml: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Cria as features de lag (defasagens temporais) e separa X e y."""
    print("\n--- 3. 🛠️ ENGENHARIA DE CARACTERÍSTICAS (Lags) ---")
    df_ml = df_diff.copy()
    
    features_to_lag = df_ml.columns
    # Usando tqdm para mostrar o progresso na criação dos lags
    for col in tqdm(features_to_lag, desc="Criando Features de Lag"):
        for i in range(1, lags_ml + 1):
            df_ml[f'{col}_Lag{i}'] = df_ml[col].shift(i)
            
    df_ml.dropna(inplace=True)
    
    features = [c for c in df_ml.columns if 'Lag' in c]
    X = df_ml[features]
    y = df_ml[TARGET_VAR]
    
    print(f"✅ {len(features)} Features de Lag criadas ({lags_ml} lags x {len(df_diff.columns)} séries).")
    
    return X, y

def split_ml_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Separa os dados de ML (com lags) em treino e teste."""
    X_treino = X.iloc[:-N_TEST_OBS]
    y_treino = y.iloc[:-N_TEST_OBS]
    X_teste = X.iloc[-N_TEST_OBS:]
    y_teste = y.iloc[-N_TEST_OBS:]
    return X_treino, y_treino, X_teste, y_teste


# ==============================================================================
# 4. SELEÇÃO RECURSIVA DE FEATURES (RFECV)
# ==============================================================================

def select_best_features(X_treino: pd.DataFrame, y_treino: pd.Series) -> List[str]:
    """Usa RFECV para selecionar o subconjunto de features que minimiza o erro."""
    print("\n--- 4. 🧠 SELEÇÃO RECURSIVA DE FEATURES (RFECV) ---")
    
    neg_mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    tscv = TimeSeriesSplit(n_splits=3)
    
    estimator = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    
    selector = RFECV(
        estimator=estimator,
        step=0.1,  
        cv=tscv,
        scoring=neg_mse_scorer,
        n_jobs=-1
    )
    
    # A barra de progresso do RFECV é difícil de implementar diretamente. 
    # Em vez disso, mostramos que o processo começou.
    print("⏳ Iniciando RFECV (pode levar alguns minutos, aguarde)...")
    selector.fit(X_treino, y_treino)
    
    best_features = X_treino.columns[selector.support_].tolist()
    
    print(f"✅ Seleção Concluída. {len(best_features)}/{len(X_treino.columns)} Features Selecionadas.")
    return best_features


# ==============================================================================
# 5. OTIMIZAÇÃO DE HIPERPARÂMETROS (Grid Search)
# ==============================================================================

def train_optimized_rf(X_treino: pd.DataFrame, y_treino: pd.Series, best_features: List[str], X_teste: pd.DataFrame) -> pd.Series:
    """Treina o Random Forest otimizando hiperparâmetros (Grid Search)."""
    print("\n--- 5. 👑 AJUSTE FINO (Grid Search nas Melhores Features) ---")
    
    X_treino_opt = X_treino[best_features]
    
    tscv = TimeSeriesSplit(n_splits=3)
    rf_params: Dict[str, List] = {
        'n_estimators': [100, 300, 500],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42), 
        rf_params,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # A barra de progresso do Grid Search também é difícil, usamos uma mensagem
    print("⏳ Iniciando Grid Search (otimizando hiperparâmetros, aguarde)...")
    rf_grid.fit(X_treino_opt, y_treino)
    
    previsao_rf = rf_grid.best_estimator_.predict(X_teste[best_features])
    previsao_rf_series = pd.Series(previsao_rf, index=X_teste.index)
    
    print(f"✅ Grid Search Concluído. Melhores Parâmetros RF: {rf_grid.best_params_}")
    
    return previsao_rf_series

# ==============================================================================
# 6. CÁLCULO DO INTERVALO DE CONFIANÇA (BOOTSTRAP)
# ==============================================================================

def calculate_bootstrap_ci(y_teste: pd.Series, previsao_rf: pd.Series, n_bootstraps: int = 1000, confidence_level: float = 0.95) -> Tuple[float, float]:
    """Calcula o Intervalo de Confiança para o RMSE usando o método Bootstrap."""
    print(f"\n--- 6. 📈 BOOTSTRAP: Calculando IC de {confidence_level*100:.0f}% ---")
    
    y_true_array = y_teste.values
    y_pred_array = previsao_rf.values
    N = len(y_teste)
    rmse_scores = []
    
    # 1. Reamostragem (1000 vezes, com reposição)
    # A barra de progresso é implementada aqui com tqdm
    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping (1000 iterações)"):
        sample_indices = np.random.choice(N, size=N, replace=True)
        y_true_sample = y_true_array[sample_indices]
        y_pred_sample = y_pred_array[sample_indices]
        sample_rmse = np.sqrt(mean_squared_error(y_true_sample, y_pred_sample))
        rmse_scores.append(sample_rmse)
        
    # 2. Determina os percentis
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    
    lower_bound = np.percentile(rmse_scores, lower_percentile * 100)
    upper_bound = np.percentile(rmse_scores, upper_percentile * 100)
    
    print(f"✅ Bootstrap Concluído em {n_bootstraps} iterações.")
    
    return lower_bound, upper_bound


# ==============================================================================
# 7. AVALIAÇÃO E VISUALIZAÇÃO
# ==============================================================================

def plot_results(y_teste: pd.Series, previsao_rf: pd.Series, rmse_rf: float, ci_lower: float, ci_upper: float, best_features: List[str]):
    """Gera e exibe o gráfico de comparação entre valores reais e previstos."""
    
    title = f'Previsão da Variação do {TARGET_VAR} com Random Forest (RMSE: {rmse_rf:.4f})'
    subtitle = f"IC 95% RMSE: [{ci_lower:.4f}, {ci_upper:.4f}] | Modelo com {len(best_features)} Features"
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(y_teste.index, y_teste, 
             label='Realizado (Variação do IBC-Br)', 
             color='#000000', 
             linewidth=3, 
             marker='o', 
             markersize=4)
             
    plt.plot(y_teste.index, previsao_rf, 
             label=f'Previsão Random Forest', 
             color='#008000', 
             linestyle='--', 
             linewidth=2, 
             marker='x', 
             markersize=6)
    
    plt.title(title + '\n' + subtitle, fontsize=14, weight='bold')
    plt.ylabel('Variação Mensal (%)', fontsize=12)
    plt.xlabel('Data', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.6, linestyle=':')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def main():
    """Função principal para execução do pipeline completo."""
    np.random.seed(42) # Fixa a semente para reprodutibilidade

    # 1. Busca e Pré-processamento
    df_treino_diff, df_teste_diff = prepare_time_series_data()
    # EXPLICAÇÃO NO TERMINAL: Mostra se os dados foram baixados corretamente e os 
    # tamanhos dos conjuntos de treino e teste.

    # 2. Otimização do Lag
    lags_ml = find_optimal_lag(df_treino_diff)
    # EXPLICAÇÃO NO TERMINAL: 'k_ar' é o número de defasagens (lags) que o modelo 
    # VAR (Vector Autoregression) sugeriu com base no critério AIC. Este lag 
    # será usado para criar as variáveis explicativas (features).
    
    # 3. Engenharia de Características (Criação de Lags)
    df_diff_full = pd.concat([df_treino_diff, df_teste_diff])
    X, y = create_lag_features(df_diff_full, lags_ml)
    # EXPLICAÇÃO NO TERMINAL: Confirma quantas variáveis preditoras (features) 
    # foram criadas, que é o produto do número de séries temporais pelo 'lags_ml'.
    
    # 4. Separação de Treino/Teste para o Modelo ML
    X_treino, y_treino, X_teste, y_teste = split_ml_data(X, y)
    
    # 5. SELEÇÃO DE FEATURES (RFECV)
    best_features = select_best_features(X_treino, y_treino)
    # EXPLICAÇÃO NO TERMINAL: Indica quantas features (defasagens) das 
    # séries macroeconômicas foram consideradas mais importantes pelo 
    # algoritmo RFECV (Random Forest) para prever o IBC-Br. Menos features 
    # tornam o modelo mais simples e menos propenso ao overfit.
    
    # 6. TREINAMENTO COM AJUSTE FINO (Grid Search)
    previsao_rf = train_optimized_rf(X_treino, y_treino, best_features, X_teste)
    # EXPLICAÇÃO NO TERMINAL: 'Melhores Parâmetros RF' são as configurações (ex: 
    # número de árvores, profundidade) que minimizam o erro do modelo na 
    # validação cruzada (Grid Search), garantindo um modelo Random Forest 
    # otimizado.
    
    # 7. Avaliação do RMSE PONTUAL
    rmse_rf = np.sqrt(mean_squared_error(y_teste, previsao_rf))
    
    # 8. CÁLCULO DO INTERVALO DE CONFIANÇA (BOOTSTRAP)
    ci_lower, ci_upper = calculate_bootstrap_ci(y_teste, previsao_rf, n_bootstraps=1000)
    # EXPLICAÇÃO NO TERMINAL: A barra de progresso mostra a reamostragem, e no 
    # final, o 'IC de 95% para o RMSE' indica a faixa de valores (intervalo) 
    # onde você pode estar 95% certo de que o erro real do seu modelo se 
    # encontra. É uma medida de incerteza da previsão.
    
    print("\n" + "="*50)
    print(f"📊 PLACAR FINAL COM INTERVALO DE CONFIANÇA")
    print(f"RMSE PONTUAL (Base de Teste): {rmse_rf:.4f}")
    # EXPLICAÇÃO FINAL: O RMSE (Raiz Quadrada do Erro Quadrático Médio) é a 
    # diferença média, em valor absoluto, entre a previsão e o valor real do 
    # IBC-Br. Quanto mais próximo de zero, melhor o modelo.
    print(f"IC de 95% para o RMSE (Bootstrap): [{ci_lower:.4f}, {ci_upper:.4f}]")
    print("="*50)
    
    # 9. Visualização
    print("\nMelhores Features Selecionadas:")
    for feature in best_features:
        print(f"- {feature}")

    plot_results(y_teste, previsao_rf, rmse_rf, ci_lower, ci_upper, best_features)


if __name__ == "__main__":
    main()