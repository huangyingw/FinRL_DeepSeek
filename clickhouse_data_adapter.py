"""
ClickHouse 数据适配器 for FinRL-DeepSeek 训练
从 ClickHouse 加载数据，替代 Hugging Face 数据集

使用方法:
    from clickhouse_data_adapter import load_training_data
    train_df, test_df = load_training_data(start_date='2018-01-01', end_date='2023-12-31')

环境变量:
    CLICKHOUSE_HOST: ClickHouse 主机 (默认: localhost)
    CLICKHOUSE_PORT: ClickHouse 端口 (默认: 9000)
    CLICKHOUSE_USER: 用户名 (默认: default)
    CLICKHOUSE_PASSWORD: 密码 (默认: 空)
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

# 添加父项目路径（仅用于引用 schema 常量）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# 表名常量（避免依赖 pkg.database.schema）
DATABASE_NAME = "trading_data"
TABLE_BARS_1D = "bars_1d"
TABLE_NEWS_SENTIMENT = "news_sentiment"


def get_full_table_name(table_name: str) -> str:
    """获取完整表名"""
    return f"{DATABASE_NAME}.{table_name}"


_client = None


def init_connection():
    """初始化 ClickHouse 连接 - 直接使用 clickhouse_driver，不依赖 EnvManager"""
    global _client
    if _client is not None:
        return

    try:
        from clickhouse_driver import Client
    except ImportError:
        raise ImportError("clickhouse_driver not installed. Run: pip install clickhouse-driver")

    # Docker 环境: 从环境变量获取配置
    host = os.environ.get('CLICKHOUSE_HOST', 'localhost')
    port = int(os.environ.get('CLICKHOUSE_PORT', 9000))
    user = os.environ.get('CLICKHOUSE_USER', 'default')
    password = os.environ.get('CLICKHOUSE_PASSWORD', '')
    database = os.environ.get('CLICKHOUSE_DATABASE', 'trading_data')

    logger.info(f"初始化 ClickHouse 连接: {host}:{port} (user: {user})")

    _client = Client(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        settings={
            'max_execution_time': 120,
            'connect_timeout': 10
        }
    )

    # 测试连接
    _client.execute("SELECT 1")
    logger.info("ClickHouse 连接成功")


class SimpleConnection:
    """简单的连接包装类，模拟 ConnectionManager 接口"""
    def __init__(self, client):
        self.client = client
        self.conn_type = 'driver'

    def execute(self, query, params=None):
        if params:
            return self.client.execute(query, params)
        return self.client.execute(query)


def get_connection():
    """获取 ClickHouse 连接"""
    global _client
    init_connection()
    return SimpleConnection(_client)

# 技术指标列表（与 Hugging Face 数据集一致）
INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma'
]


def get_ohlc_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """从 ClickHouse 获取 OHLC 数据（每日聚合）"""
    conn = get_connection()
    table = get_full_table_name(TABLE_BARS_1D)

    symbols_str = ", ".join([f"'{s}'" for s in symbols])

    # 使用聚合函数获取每日唯一数据（bars_1d 表可能有多条同日记录）
    query = f"""
        SELECT
            symbol as tic,
            toDate(timestamp) as date,
            argMin(open, timestamp) as open,
            max(high) as high,
            min(low) as low,
            argMax(close, timestamp) as close,
            sum(volume) as volume
        FROM {table}
        WHERE symbol IN ({symbols_str})
          AND timestamp >= '{start_date}'
          AND timestamp <= '{end_date}'
        GROUP BY symbol, toDate(timestamp)
        ORDER BY date, symbol
    """

    result = conn.execute(query)
    df = pd.DataFrame(result, columns=['tic', 'date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return df


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


def get_sentiment_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """从 ClickHouse 获取情感数据

    news_sentiment 表结构:
    - symbol, published_at, sentiment ('positive', 'negative', 'neutral')

    将 sentiment 字符串转换为数值: positive=1, neutral=0, negative=-1
    """
    conn = get_connection()
    table = get_full_table_name(TABLE_NEWS_SENTIMENT)

    symbols_str = ", ".join([f"'{s}'" for s in symbols])

    # 使用 published_at 代替 timestamp, 将 sentiment 字符串转换为数值
    query = f"""
        SELECT
            symbol as tic,
            toDate(published_at) as date,
            avg(
                CASE sentiment
                    WHEN 'positive' THEN 1.0
                    WHEN 'negative' THEN -1.0
                    ELSE 0.0
                END
            ) as avg_sentiment,
            count() as news_count
        FROM {table}
        WHERE symbol IN ({symbols_str})
          AND published_at >= '{start_date}'
          AND published_at <= '{end_date}'
        GROUP BY symbol, toDate(published_at)
        ORDER BY date, symbol
    """

    try:
        result = conn.execute(query)
        df = pd.DataFrame(result, columns=['tic', 'date', 'avg_sentiment', 'news_count'])
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        logger.warning(f"无法获取情感数据: {e}，将使用模拟数据")
        return pd.DataFrame(columns=['tic', 'date', 'avg_sentiment', 'news_count'])


def convert_sentiment_to_score(sentiment: float) -> int:
    """
    将情感分数 (-1.0 到 1.0) 转换为 1-5 评分
    1 = 非常消极, 3 = 中性, 5 = 非常积极
    """
    if pd.isna(sentiment):
        return 0  # 0 表示无数据
    # 线性映射: [-1, 1] -> [1, 5]
    score = int((sentiment + 1) * 2 + 1)
    return max(1, min(5, score))


def calculate_risk_score(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    计算风险评分 (1-5)
    基于波动率和价格变化
    1 = 低风险, 5 = 高风险
    """
    result_dfs = []

    for tic in df['tic'].unique():
        tic_df = df[df['tic'] == tic].copy().sort_values('date')

        # 计算收益率波动
        tic_df['return'] = tic_df['close'].pct_change()
        tic_df['volatility'] = tic_df['return'].rolling(window=window).std()

        # 波动率分位数转换为风险评分
        vol_percentile = tic_df['volatility'].rank(pct=True)
        tic_df['llm_risk'] = (vol_percentile * 4 + 1).round().fillna(3).astype(int)
        tic_df['llm_risk'] = tic_df['llm_risk'].clip(1, 5)

        result_dfs.append(tic_df)

    return pd.concat(result_dfs, ignore_index=True)


def get_nasdaq100_symbols() -> List[str]:
    """获取 NASDAQ-100 股票列表"""
    # 常见的 NASDAQ-100 成分股
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
    start_date: str = '2018-01-01',
    end_date: str = '2023-12-31',
    test_ratio: float = 0.2,
    symbols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从 ClickHouse 加载训练和测试数据

    Args:
        start_date: 开始日期
        end_date: 结束日期
        test_ratio: 测试集比例
        symbols: 股票列表，默认使用 NASDAQ-100

    Returns:
        train_df, test_df: 训练和测试数据
    """
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
        df['llm_sentiment'] = 0  # 无情感数据

    # 4. 计算风险评分
    df = calculate_risk_score(df)

    # 5. 填充缺失值
    df = df.fillna({
        'llm_sentiment': 0,
        'llm_risk': 3,  # 中性风险
        'macd': 0, 'boll_ub': 0, 'boll_lb': 0,
        'rsi_30': 50, 'cci_30': 0, 'dx_30': 0,
        'close_30_sma': 0, 'close_60_sma': 0
    })

    # 6. 删除包含 NaN 的行（主要是初始窗口期）
    df = df.dropna()

    # 7. 确保每日股票数量一致（FinRL 环境要求）
    date_counts = df.groupby('date')['tic'].count()
    valid_dates = date_counts[date_counts == date_counts.max()].index.tolist()
    df = df[df['date'].isin(valid_dates)]

    # 只保留所有日期都有数据的股票
    n_dates = df['date'].nunique()
    tic_counts = df.groupby('tic')['date'].count()
    valid_tics = tic_counts[tic_counts == n_dates].index.tolist()
    df = df[df['tic'].isin(valid_tics)]

    logger.info(f"数据清洗后: {len(df)} 行, {df['tic'].nunique()} 只股票, {df['date'].nunique()} 天")

    # 7. 按日期划分训练/测试集
    unique_dates = sorted(df['date'].unique())

    if test_ratio <= 0 or len(unique_dates) == 0:
        # 无测试集，全部作为训练集
        train_df = df.reset_index(drop=True)
        test_df = pd.DataFrame(columns=df.columns)
    else:
        split_idx = int(len(unique_dates) * (1 - test_ratio))
        split_idx = min(split_idx, len(unique_dates) - 1)  # 防止越界
        split_date = unique_dates[split_idx]

        train_df = df[df['date'] < split_date].reset_index(drop=True)
        test_df = df[df['date'] >= split_date].reset_index(drop=True)

    logger.info(f"训练集: {len(train_df)} 行, 测试集: {len(test_df)} 行")
    if len(train_df) > 0:
        logger.info(f"训练集日期: {train_df['date'].min()} ~ {train_df['date'].max()}")
    if len(test_df) > 0:
        logger.info(f"测试集日期: {test_df['date'].min()} ~ {test_df['date'].max()}")

    return train_df, test_df


def create_mock_data_for_testing(n_stocks: int = 10, n_days: int = 500) -> pd.DataFrame:
    """
    创建模拟数据用于测试（当 ClickHouse 不可用时）
    """
    np.random.seed(42)
    symbols = [f'TEST{i:02d}' for i in range(n_stocks)]
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    data = []
    for symbol in symbols:
        base_price = np.random.uniform(50, 500)
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = base_price * np.cumprod(1 + returns)

        for i, (date, price) in enumerate(zip(dates, prices)):
            data.append({
                'tic': symbol,
                'date': date.strftime('%Y-%m-%d'),
                'open': price * (1 + np.random.uniform(-0.01, 0.01)),
                'high': price * (1 + np.random.uniform(0, 0.02)),
                'low': price * (1 - np.random.uniform(0, 0.02)),
                'close': price,
                'volume': int(np.random.uniform(1e6, 1e8)),
                'llm_sentiment': np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.1, 0.1, 0.2, 0.15, 0.15]),
                'llm_risk': np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1]),
            })

    df = pd.DataFrame(data)

    # 添加技术指标
    for indicator in INDICATORS:
        df[indicator] = np.random.randn(len(df))

    return df


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    try:
        train_df, test_df = load_training_data(
            start_date='2020-01-01',
            end_date='2023-12-31',
            symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        )
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"\nColumns: {train_df.columns.tolist()}")
        print(f"\nSample:\n{train_df.head()}")
    except Exception as e:
        print(f"ClickHouse 连接失败: {e}")
        print("使用模拟数据测试...")
        mock_df = create_mock_data_for_testing()
        print(f"Mock data shape: {mock_df.shape}")
        print(f"\nColumns: {mock_df.columns.tolist()}")
