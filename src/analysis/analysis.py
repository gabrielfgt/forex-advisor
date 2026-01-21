"""
Módulo de análise técnica e classificação de cenários de mercado.
Calcula indicadores técnicos e classifica o cenário atual em 4 categorias.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


def calculate_sma(df, period):
    """Calcula Simple Moving Average."""
    return df['Close'].rolling(window=period).mean()


def calculate_rsi(df, period=14):
    """Calcula Relative Strength Index (RSI)."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(df, period=20, num_std=2):
    """Calcula Bandas de Bollinger."""
    sma = calculate_sma(df, period)
    std = df['Close'].rolling(window=period).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return upper_band, sma, lower_band


def calculate_volatility(df, period=20):
    """Calcula volatilidade histórica (desvio padrão dos retornos)."""
    returns = df['Close'].pct_change()
    volatility = returns.rolling(window=period).std() * np.sqrt(252) * 100  # Anualizada em %
    return volatility


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calcula MACD (Moving Average Convergence Divergence)."""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_support_resistance(df, window=20):
    """Identifica níveis de suporte e resistência usando mínimos e máximos locais."""
    support = df['Low'].rolling(window=window, center=True).min()
    resistance = df['High'].rolling(window=window, center=True).max()
    
    return support, resistance


def calculate_all_indicators(df):
    """
    Calcula todos os indicadores técnicos.
    
    Args:
        df (pd.DataFrame): DataFrame com dados OHLC
    
    Returns:
        pd.DataFrame: DataFrame com indicadores adicionados
    """
    df = df.copy()
    
    # Médias Móveis
    df['SMA_20'] = calculate_sma(df, 20)
    df['SMA_50'] = calculate_sma(df, 50)
    df['SMA_200'] = calculate_sma(df, 200)
    
    # RSI
    df['RSI'] = calculate_rsi(df, 14)
    
    # Bandas de Bollinger
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df, 20, 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volatilidade
    df['Volatility'] = calculate_volatility(df, 20)
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df)
    
    # Suporte e Resistência
    df['Support'], df['Resistance'] = calculate_support_resistance(df, 20)
    
    # Retornos
    df['Returns'] = df['Close'].pct_change()
    
    # Tendência (direção da média móvel)
    df['Trend_20'] = df['SMA_20'].diff()
    df['Trend_50'] = df['SMA_50'].diff()
    
    return df


def classify_heuristic(df):
    """
    Classifica o cenário atual usando heurística baseada em regras.
    
    Args:
        df (pd.DataFrame): DataFrame com indicadores calculados
    
    Returns:
        dict: Dicionário com classificação, confiança e explicação
    """
    if len(df) == 0:
        return {
            'classification': 'Neutro',
            'confidence': 0.0,
            'explanation': 'Dados insuficientes para classificação'
        }
    
    latest = df.iloc[-1]
    
    required_indicators = ['SMA_20', 'SMA_50', 'RSI', 'BB_Width', 'Volatility']
    if any(pd.isna(latest[ind]) for ind in required_indicators):
        return {
            'classification': 'Neutro',
            'confidence': 0.0,
            'explanation': 'Indicadores insuficientes para classificação'
        }
    
    price = latest['Close']
    sma_20 = latest['SMA_20']
    sma_50 = latest['SMA_50']
    rsi = latest['RSI']
    bb_width = latest['BB_Width']
    volatility = latest['Volatility']
    bb_position = latest['BB_Position']
    
    volatility_percentile = (df['Volatility'].dropna() < volatility).sum() / len(df['Volatility'].dropna()) * 100
    
    scores = {
        'Tendência de Alta': 0,
        'Tendência de Baixa': 0,
        'Alta Volatilidade': 0,
        'Neutro': 0
    }
    
    # Regras para Tendência de Alta
    if price > sma_20:
        scores['Tendência de Alta'] += 1
    if price > sma_50:
        scores['Tendência de Alta'] += 1
    if 50 <= rsi <= 70:
        scores['Tendência de Alta'] += 1
    if latest.get('Trend_20', 0) > 0:
        scores['Tendência de Alta'] += 1
    
    # Regras para Tendência de Baixa
    if price < sma_20:
        scores['Tendência de Baixa'] += 1
    if price < sma_50:
        scores['Tendência de Baixa'] += 1
    if 30 <= rsi <= 50:
        scores['Tendência de Baixa'] += 1
    if latest.get('Trend_20', 0) < 0:
        scores['Tendência de Baixa'] += 1
    
    # Regras para Alta Volatilidade
    if volatility_percentile > 75:
        scores['Alta Volatilidade'] += 2
    if bb_width > df['BB_Width'].quantile(0.75):
        scores['Alta Volatilidade'] += 1
    if bb_position < 0.2 or bb_position > 0.8:
        scores['Alta Volatilidade'] += 1
    
    # Se nenhuma categoria se destacou, é Neutro
    max_score = max(scores.values())
    if max_score == 0:
        scores['Neutro'] = 1
        classification = 'Neutro'
        confidence = 0.5
    else:
        classification = max(scores, key=scores.get)
        confidence = min(max_score / 4.0, 1.0)
    
    explanation_parts = []
    if classification == 'Tendência de Alta':
        explanation_parts.append(f"Preço ({price:.4f}) acima das médias móveis")
        explanation_parts.append(f"RSI em {rsi:.2f} (zona neutra-alta)")
    elif classification == 'Tendência de Baixa':
        explanation_parts.append(f"Preço ({price:.4f}) abaixo das médias móveis")
        explanation_parts.append(f"RSI em {rsi:.2f} (zona neutra-baixa)")
    elif classification == 'Alta Volatilidade':
        explanation_parts.append(f"Volatilidade no percentil {volatility_percentile:.1f}%")
        explanation_parts.append(f"Bandas de Bollinger expandidas")
    else:
        explanation_parts.append("Indicadores mistos, sem tendência clara")
    
    explanation = "; ".join(explanation_parts)
    
    return {
        'classification': classification,
        'confidence': confidence,
        'explanation': explanation,
        'scores': scores
    }


def get_feature_importance(df):
    """
    Treina um modelo auxiliar (Random Forest) para explicabilidade das features.
    
    Args:
        df (pd.DataFrame): DataFrame com indicadores calculados
    
    Returns:
        dict: Dicionário com importância das features
    """
    try:
        feature_cols = [
            'SMA_20', 'SMA_50', 'RSI', 'BB_Width', 'BB_Position',
            'Volatility', 'MACD', 'MACD_Histogram', 'Returns'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        df_clean = df[available_features + ['Close']].dropna()
        
        if len(df_clean) < 50:
            logger.warning("Dados insuficientes para treinar modelo de explicabilidade")
            return {}
        
        targets = []
        for idx in range(len(df_clean)):
            row_df = df_clean.iloc[:idx+1]
            if len(row_df) > 0:
                classification = classify_heuristic(row_df)['classification']
                targets.append(classification)
            else:
                targets.append('Neutro')
        
        df_clean['Target'] = targets
        
        X = df_clean[available_features]
        y = df_clean['Target']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_scaled, y)
        
        feature_importance = dict(zip(available_features, rf.feature_importances_))
        
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    except Exception as e:
        logger.warning(f"Erro ao calcular importância de features: {str(e)}")
        return {}


def analyze_market(df):
    """
    Função principal que executa análise completa do mercado.
    
    Args:
        df (pd.DataFrame): DataFrame com dados OHLC
    
    Returns:
        dict: Dicionário completo com análise, classificação e explicabilidade
    """
    df_with_indicators = calculate_all_indicators(df)
    
    classification = classify_heuristic(df_with_indicators)
    
    feature_importance = get_feature_importance(df_with_indicators)
    
    latest = df_with_indicators.iloc[-1]
    indicators_summary = {
        'price': float(latest['Close']),
        'sma_20': float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
        'sma_50': float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None,
        'rsi': float(latest['RSI']) if not pd.isna(latest['RSI']) else None,
        'volatility': float(latest['Volatility']) if not pd.isna(latest['Volatility']) else None,
        'bb_width': float(latest['BB_Width']) if not pd.isna(latest['BB_Width']) else None,
        'bb_position': float(latest['BB_Position']) if not pd.isna(latest['BB_Position']) else None,
    }
    
    return {
        'classification': classification['classification'],
        'confidence': classification['confidence'],
        'explanation': classification['explanation'],
        'feature_contributions': feature_importance,
        'indicators_summary': indicators_summary,
        'latest_data': latest.to_dict()
    }


def display_dataframe_structure(df_raw, df_with_indicators):
    """
    Exibe a estrutura do DataFrame em formato tabular.
    
    Args:
        df_raw: DataFrame com dados brutos OHLC
        df_with_indicators: DataFrame com indicadores calculados
    """
    print("\n" + "="*100)
    print("ESTRUTURA DO DATAFRAME - DADOS BRUTOS (OHLC)")
    print("="*100)
    print(f"\nShape: {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas")
    print(f"\nColunas: {list(df_raw.columns)}")
    print("\nPrimeiras 5 linhas:")
    print(df_raw.head().to_string())
    print("\nÚltimas 5 linhas:")
    print(df_raw.tail().to_string())
    print("\nInformações do DataFrame:")
    print(df_raw.info())
    print("\nEstatísticas Descritivas:")
    print(df_raw.describe().to_string())
    
    print("\n" + "="*100)
    print("ESTRUTURA DO DATAFRAME - COM INDICADORES TÉCNICOS")
    print("="*100)
    print(f"\nShape: {df_with_indicators.shape[0]} linhas × {df_with_indicators.shape[1]} colunas")
    print(f"\nColunas ({len(df_with_indicators.columns)}):")
    for i, col in enumerate(df_with_indicators.columns, 1):
        print(f"  {i:2d}. {col}")
    
    categories = {
        'Dados Originais': ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
        'Médias Móveis': ['SMA_20', 'SMA_50', 'SMA_200'],
        'RSI': ['RSI'],
        'Bandas de Bollinger': ['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position'],
        'Volatilidade': ['Volatility'],
        'MACD': ['MACD', 'MACD_Signal', 'MACD_Histogram'],
        'Suporte/Resistência': ['Support', 'Resistance'],
        'Retornos e Tendências': ['Returns', 'Trend_20', 'Trend_50']
    }
    
    print("\n" + "-"*100)
    print("COLUNAS POR CATEGORIA:")
    print("-"*100)
    for category, cols in categories.items():
        available_cols = [c for c in cols if c in df_with_indicators.columns]
        if available_cols:
            print(f"\n{category}:")
            for col in available_cols:
                print(f"  - {col}")
    
    print("\n" + "-"*100)
    print("ÚLTIMAS 10 LINHAS COM INDICADORES (DADOS MAIS RECENTES):")
    print("-"*100)
    
    # Selecionar colunas principais para visualização
    display_cols = ['Date', 'Close', 'SMA_20', 'SMA_50', 'RSI', 'BB_Width', 
                    'Volatility', 'MACD', 'Returns', 'Trend_20']
    available_display_cols = [c for c in display_cols if c in df_with_indicators.columns]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}' if abs(x) < 1000 else f'{x:.2f}')
    
    print(df_with_indicators[available_display_cols].tail(10).to_string())
    
    print("\n" + "-"*100)
    print("VALORES ATUAIS (ÚLTIMA LINHA):")
    print("-"*100)
    latest = df_with_indicators.iloc[-1]
    for col in df_with_indicators.columns:
        value = latest[col]
        if pd.notna(value):
            if isinstance(value, (int, float)):
                print(f"{col:25s}: {value:15.4f}")
            else:
                print(f"{col:25s}: {str(value):15s}")
        else:
            print(f"{col:25s}: {'NaN':15s}")


if __name__ == "__main__":
    import sys
    import os
    import importlib.util
    
    forex_module_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "forex-scrapping.py")
    spec = importlib.util.spec_from_file_location("forex_scrapping", forex_module_path)
    forex_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forex_module)
    fetch_forex_data = forex_module.fetch_forex_data
    
    logging.basicConfig(level=logging.INFO)
    
    data = fetch_forex_data(years=1)
    
    data_with_indicators = calculate_all_indicators(data)
    
    display_dataframe_structure(data, data_with_indicators)
    
    analysis = analyze_market(data)
    
    print("\n" + "="*100)
    print("ANÁLISE DE MERCADO")
    print("="*100)
    print(f"\nClassificação: {analysis['classification']}")
    print(f"Confiança: {analysis['confidence']:.2%}")
    print(f"Explicação: {analysis['explanation']}")
    print("\n=== Importância das Features ===")
    for feature, importance in analysis['feature_contributions'].items():
        print(f"{feature}: {importance:.4f}")
    print("\n=== Indicadores Atuais ===")
    for key, value in analysis['indicators_summary'].items():
        print(f"{key}: {value}")
