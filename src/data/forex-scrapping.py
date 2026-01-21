"""
Módulo de coleta de dados OHLC para o par BRL/USD.
Busca dados históricos dos últimos 5 anos usando yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def fetch_forex_data(years=5):
    """
    Busca dados OHLC históricos do par BRL/USD.
    
    Args:
        years (int): Número de anos de dados históricos a buscar (padrão: 5)
    
    Returns:
        pd.DataFrame: DataFrame com colunas Date, Open, High, Low, Close, Volume
    """
    try:
        ticker = "BRL=X"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        logger.info(f"Buscando dados de {ticker} de {start_date.date()} até {end_date.date()}")
        
        forex_ticker = yf.Ticker(ticker)
        df = forex_ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"Nenhum dado encontrado para {ticker}")
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Colunas faltando no DataFrame: {missing_columns}")
        
        df = df.reset_index()
        if 'Date' not in df.columns and 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        
        df = validate_data(df)
        
        logger.info(f"Dados coletados com sucesso: {len(df)} registros")
        return df
    
    except Exception as e:
        logger.error(f"Erro ao buscar dados de forex: {str(e)}")
        raise


def validate_data(df):
    """
    Valida e limpa os dados OHLC.
    
    Args:
        df (pd.DataFrame): DataFrame com dados OHLC
    
    Returns:
        pd.DataFrame: DataFrame validado e limpo
    """
    initial_len = len(df)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    if len(df) < initial_len:
        logger.warning(f"Removidas {initial_len - len(df)} linhas com valores NaN")
    
    invalid_rows = (
        (df['High'] < df['Low']) |
        (df['High'] < df['Open']) |
        (df['High'] < df['Close']) |
        (df['Low'] > df['Open']) |
        (df['Low'] > df['Close'])
    )
    
    if invalid_rows.any():
        num_invalid = invalid_rows.sum()
        logger.warning(f"Encontradas {num_invalid} linhas com lógica OHLC inválida. Removendo...")
        df = df[~invalid_rows]
    
    for col in ['Open', 'High', 'Low', 'Close']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        if outliers.any():
            logger.warning(f"Encontrados {outliers.sum()} outliers em {col} (não removidos, apenas logados)")
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def get_latest_data(df, days=1):
    """
    Retorna os dados mais recentes do DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame com dados históricos
        days (int): Número de dias mais recentes a retornar
    
    Returns:
        pd.DataFrame: DataFrame com os dados mais recentes
    """
    return df.tail(days).copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = fetch_forex_data(years=5)
    print(f"\nDados coletados:")
    print(data.head())
    print(f"\nShape: {data.shape}")
    print(f"\nÚltimos 5 registros:")
    print(data.tail())
