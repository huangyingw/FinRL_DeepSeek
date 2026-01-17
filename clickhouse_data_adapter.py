"""
ClickHouse 数据适配器 for FinRL-DeepSeek 训练
从 ClickHouse 加载数据，替代 Hugging Face 数据集

使用方法:
    from clickhouse_data_adapter import load_training_data
    train_df, test_df = load_training_data()

配置来源:
    复用父项目 pkg.database.repository
    - 非敏感配置: 从 config/settings.yaml 读取 (通过 EnvManager)
    - 敏感数据: 通过 EnvManager → Doppler SDK 获取
"""

import os
import sys
import logging
from datetime import datetime
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

# 添加父项目路径，复用 pkg.database 封装层
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from pkg.database import repository as db

logger = logging.getLogger(__name__)

# 技术指标列表（与 Hugging Face 数据集一致）
INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma'
]


def _ensure_db_initialized():
    """确保数据库已初始化"""
    if not db.is_initialized():
        db.initialize()
        logger.info("数据库已初始化 (通过 pkg.database.repository)")


def get_ohlc_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """从 ClickHouse 获取 OHLC 数据（每日聚合）

    使用 pkg.database.repository 封装层，不直接写 SQL
    """
    _ensure_db_initialized()
    df = db.get_bars_df_by_date(symbols, '1d', start_date, end_date)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return df


def get_sentiment_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """从 ClickHouse 获取情感数据

    使用 pkg.database.repository 封装层，不直接写 SQL
    """
    _ensure_db_initialized()
    try:
        df = db.get_sentiment_df(symbols, start_date, end_date)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        logger.warning(f"无法获取情感数据: {e}")
        return pd.DataFrame(columns=['tic', 'date', 'avg_sentiment'])


def get_available_date_range() -> Tuple[str, str]:
    """从数据库查询可用数据的日期范围

    使用 pkg.database.repository 封装层，不直接写 SQL
    """
    _ensure_db_initialized()
    try:
        min_date, max_date = db.get_date_range('1d')
        start = min_date.strftime('%Y-%m-%d')
        end = max_date.strftime('%Y-%m-%d')
        logger.info(f"数据库日期范围: {start} ~ {end}")
        return start, end
    except Exception as e:
        logger.warning(f"获取日期范围失败: {e}，使用默认值")
        return '2018-01-01', '2023-12-31'


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    import ta

    result_dfs = []

    for tic in df['tic'].unique():
        tic_df = df[df['tic'] == tic].copy().sort_values('date')

        # MACD
        macd = ta.trend.MACD(tic_df['close'])
        tic_df['macd'] = macd.macd()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(tic_df['close'], window=20)
        tic_df['boll_ub'] = bb.bollinger_hband()
        tic_df['boll_lb'] = bb.bollinger_lband()

        # RSI
        tic_df['rsi_30'] = ta.momentum.RSIIndicator(tic_df['close'], window=30).rsi()

        # CCI
        tic_df['cci_30'] = ta.trend.CCIIndicator(
            tic_df['high'], tic_df['low'], tic_df['close'], window=30
        ).cci()

        # DX (ADX)
        adx = ta.trend.ADXIndicator(tic_df['high'], tic_df['low'], tic_df['close'], window=30)
        tic_df['dx_30'] = adx.adx()

        # SMA
        tic_df['close_30_sma'] = tic_df['close'].rolling(window=30).mean()
        tic_df['close_60_sma'] = tic_df['close'].rolling(window=60).mean()

        result_dfs.append(tic_df)

    return pd.concat(result_dfs, ignore_index=True)


def convert_sentiment_to_score(sentiment: float) -> int:
    """将情感分数 (-1.0 到 1.0) 转换为 1-5 评分"""
    if pd.isna(sentiment):
        return 0
    score = int((sentiment + 1) * 2 + 1)
    return max(1, min(5, score))


def calculate_risk_score(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """计算风险评分 (1-5)"""
    result_dfs = []

    for tic in df['tic'].unique():
        tic_df = df[df['tic'] == tic].copy().sort_values('date')

        tic_df['return'] = tic_df['close'].pct_change()
        tic_df['volatility'] = tic_df['return'].rolling(window=window).std()

        vol_percentile = tic_df['volatility'].rank(pct=True)
        tic_df['llm_risk'] = (vol_percentile * 4 + 1).round().fillna(3).astype(int)
        tic_df['llm_risk'] = tic_df['llm_risk'].clip(1, 5)

        result_dfs.append(tic_df)

    return pd.concat(result_dfs, ignore_index=True)


def get_nasdaq100_symbols() -> List[str]:
    """获取 NASDAQ-100 股票列表"""
    return [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AVGO', 'COST', 'NFLX',
        'AMD', 'ADBE', 'QCOM', 'PEP', 'CSCO', 'INTC', 'TMUS', 'CMCSA', 'TXN', 'AMGN',
        'INTU', 'HON', 'ISRG', 'BKNG', 'AMAT', 'VRTX', 'SBUX', 'MDLZ', 'GILD', 'ADI',
        'ADP', 'REGN', 'LRCX', 'PANW', 'MU', 'KLAC', 'SNPS', 'CDNS', 'MELI', 'PYPL',
        'CTAS', 'MAR', 'CSX', 'ORLY', 'NXPI', 'ASML', 'WDAY', 'MNST', 'CHTR', 'MRVL',
        'PCAR', 'ADSK', 'FTNT', 'AEP', 'CPRT', 'ROST', 'KDP', 'PAYX', 'ODFL', 'EXC',
        'LULU', 'AZN', 'KHC', 'CEG', 'MRNA', 'FAST', 'DXCM', 'BKR', 'EA', 'VRSK',
        'CTSH', 'XEL', 'IDXX', 'GEHC', 'CSGP', 'TTD', 'ANSS', 'ON', 'FANG', 'ZS'
    ]


def load_training_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    test_ratio: float = 0.2,
    symbols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从 ClickHouse 加载训练和测试数据

    Args:
        start_date: 开始日期 (默认: 数据库最早日期)
        end_date: 结束日期 (默认: 数据库最新日期)
        test_ratio: 测试集比例 (默认: 0.2)
        symbols: 股票列表，默认使用 NASDAQ-100

    Returns:
        train_df, test_df: 训练和测试数据
    """
    # 动态获取日期范围
    if start_date is None or end_date is None:
        db_start, db_end = get_available_date_range()
        if start_date is None:
            start_date = db_start
        if end_date is None:
            end_date = db_end

    if symbols is None:
        symbols = get_nasdaq100_symbols()

    logger.info(f"从 ClickHouse 加载数据: {start_date} 到 {end_date}, {len(symbols)} 只股票")

    # 1. 获取 OHLC 数据
    ohlc_df = get_ohlc_data(symbols, start_date, end_date)
    logger.info(f"OHLC 数据: {len(ohlc_df)} 行")

    if ohlc_df.empty:
        raise ValueError("无法从 ClickHouse 获取 OHLC 数据，请确保数据库中有数据")

    # 2. 计算技术指标
    df = calculate_technical_indicators(ohlc_df)
    logger.info(f"技术指标计算完成")

    # 3. 获取情感数据
    sentiment_df = get_sentiment_data(symbols, start_date, end_date)
    if not sentiment_df.empty:
        df = df.merge(sentiment_df, on=['tic', 'date'], how='left')
        df['llm_sentiment'] = df['avg_sentiment'].apply(convert_sentiment_to_score)
    else:
        df['llm_sentiment'] = 0

    # 4. 计算风险评分
    df = calculate_risk_score(df)

    # 5. 填充缺失值
    df = df.fillna({
        'llm_sentiment': 0,
        'llm_risk': 3,
        'macd': 0, 'boll_ub': 0, 'boll_lb': 0,
        'rsi_30': 50, 'cci_30': 0, 'dx_30': 0,
        'close_30_sma': 0, 'close_60_sma': 0
    })

    # 6. 删除包含 NaN 的行
    df = df.dropna()

    # 7. 确保每日股票数量一致
    date_counts = df.groupby('date')['tic'].count()
    valid_dates = date_counts[date_counts == date_counts.max()].index.tolist()
    df = df[df['date'].isin(valid_dates)]

    n_dates = df['date'].nunique()
    tic_counts = df.groupby('tic')['date'].count()
    valid_tics = tic_counts[tic_counts == n_dates].index.tolist()
    df = df[df['tic'].isin(valid_tics)]

    logger.info(f"数据清洗后: {len(df)} 行, {df['tic'].nunique()} 只股票, {df['date'].nunique()} 天")

    # 8. 按日期划分训练/测试集
    unique_dates = sorted(df['date'].unique())

    if test_ratio <= 0 or len(unique_dates) == 0:
        train_df = df.reset_index(drop=True)
        test_df = pd.DataFrame(columns=df.columns)
    else:
        split_idx = int(len(unique_dates) * (1 - test_ratio))
        split_idx = min(split_idx, len(unique_dates) - 1)
        split_date = unique_dates[split_idx]

        train_df = df[df['date'] < split_date].reset_index(drop=True)
        test_df = df[df['date'] >= split_date].reset_index(drop=True)

    logger.info(f"训练集: {len(train_df)} 行, 测试集: {len(test_df)} 行")
    return train_df, test_df


def create_mock_data_for_testing() -> pd.DataFrame:
    """创建模拟数据用于测试"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    data = []
    for symbol in symbols:
        base_price = 100 + np.random.randn() * 20
        for date in dates:
            price = base_price * (1 + np.random.randn() * 0.02)
            data.append({
                'tic': symbol,
                'date': date.strftime('%Y-%m-%d'),
                'open': price,
                'high': price * 1.01,
                'low': price * 0.99,
                'close': price,
                'volume': int(1000000 * (1 + np.random.rand())),
                'macd': np.random.randn() * 0.5,
                'boll_ub': price * 1.02,
                'boll_lb': price * 0.98,
                'rsi_30': 50 + np.random.randn() * 10,
                'cci_30': np.random.randn() * 50,
                'dx_30': 20 + np.random.rand() * 30,
                'close_30_sma': price,
                'close_60_sma': price,
                'llm_sentiment': np.random.choice([1, 2, 3, 4, 5]),
                'llm_risk': np.random.choice([1, 2, 3, 4, 5])
            })

    return pd.DataFrame(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        train_df, test_df = load_training_data()
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"\nColumns: {train_df.columns.tolist()}")
        print(f"\nSample:\n{train_df.head()}")
    except Exception as e:
        print(f"ClickHouse 连接失败: {e}")
        print("使用模拟数据测试...")
        mock_df = create_mock_data_for_testing()
        print(f"Mock data shape: {mock_df.shape}")
