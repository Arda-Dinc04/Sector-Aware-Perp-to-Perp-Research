"""
GMX v2 data collector for perpetual futures data.
Collects funding/borrow rates, mark prices, index prices, and market data.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
import logging
from .base_collector import BaseDataCollector

logger = logging.getLogger(__name__)

class GmxCollector(BaseDataCollector):
    """GMX v2 data collector."""
    
    def __init__(self, data_dir: str = "raw_data"):
        super().__init__("gmx", data_dir)
        # GMX v2 uses The Graph subgraph for data
        self.subgraph_url = "https://api.thegraph.com/subgraphs/name/gmx-io/gmx-arbitrum-stats"
        self.rate_limit_delay = 0.2  # 200ms between requests
        
    def get_taker_fees(self) -> float:
        """GMX taker fees: 0.1% = 10 bps"""
        return 10.0
    
    def get_maker_fees(self) -> float:
        """GMX doesn't have traditional maker fees (LP model)"""
        return 0.0
    
    def get_borrow_fees(self) -> float:
        """GMX borrow fees vary by asset, use average"""
        return 5.0  # 5 bps per day average
    
    def fetch_markets_list(self) -> List[Dict[str, Any]]:
        """Fetch list of available markets from GMX."""
        try:
            # GMX v2 markets query
            query = """
            {
                tokens(first: 100, where: {isStableToken: false}) {
                    id
                    symbol
                    name
                    decimals
                    isStableToken
                    isShortable
                }
            }
            """
            
            response = requests.post(
                self.subgraph_url,
                json={'query': query},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            markets = []
            
            for token in data.get('data', {}).get('tokens', []):
                if not token.get('isStableToken', True) and token.get('isShortable', False):
                    markets.append({
                        'symbol': token['symbol'],
                        'id': token['id'],
                        'name': token['name'],
                        'decimals': int(token['decimals']),
                        'is_shortable': token.get('isShortable', False)
                    })
            
            logger.info(f"Found {len(markets)} tradeable tokens on GMX")
            return markets
            
        except Exception as e:
            logger.error(f"Failed to fetch GMX markets: {str(e)}")
            return []
    
    def fetch_funding_rates(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch funding/borrow rates for a symbol from GMX."""
        try:
            # Convert times to timestamps
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            
            # GMX uses funding/borrow rates based on OI skew
            # We need to fetch the funding rate history
            query = """
            query($token: String!, $from: Int!, $to: Int!) {
                fundingRateUpdates(
                    first: 1000,
                    where: {
                        token: $token,
                        timestamp_gte: $from,
                        timestamp_lte: $to
                    },
                    orderBy: timestamp,
                    orderDirection: asc
                ) {
                    id
                    token
                    timestamp
                    fundingRate
                    borrowRate
                }
            }
            """
            
            variables = {
                "token": symbol,
                "from": int(start_dt.timestamp()),
                "to": int(end_dt.timestamp())
            }
            
            response = requests.post(
                self.subgraph_url,
                json={'query': query, 'variables': variables},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            funding_data = []
            
            for item in data.get('data', {}).get('fundingRateUpdates', []):
                timestamp = pd.to_datetime(int(item['timestamp']), unit='s')
                if start_dt <= timestamp <= end_dt:
                    # GMX funding rate is the net rate (funding - borrow)
                    funding_rate = float(item.get('fundingRate', 0)) - float(item.get('borrowRate', 0))
                    funding_data.append({
                        'timestamp': timestamp,
                        'funding_rate': funding_rate,
                        'borrow_rate': float(item.get('borrowRate', 0))
                    })
            
            if not funding_data:
                logger.warning(f"No funding rate data found for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(funding_data)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            
            logger.info(f"Fetched {len(df)} funding rate records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch GMX funding rates for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_price_data(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch price data for a symbol from GMX."""
        try:
            # Convert times to timestamps
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            
            # GMX price data query
            query = """
            query($token: String!, $from: Int!, $to: Int!) {
                priceUpdates(
                    first: 1000,
                    where: {
                        token: $token,
                        timestamp_gte: $from,
                        timestamp_lte: $to
                    },
                    orderBy: timestamp,
                    orderDirection: asc
                ) {
                    id
                    token
                    timestamp
                    price
                    minPrice
                    maxPrice
                }
            }
            """
            
            variables = {
                "token": symbol,
                "from": int(start_dt.timestamp()),
                "to": int(end_dt.timestamp())
            }
            
            response = requests.post(
                self.subgraph_url,
                json={'query': query, 'variables': variables},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            price_data = []
            
            for item in data.get('data', {}).get('priceUpdates', []):
                timestamp = pd.to_datetime(int(item['timestamp']), unit='s')
                if start_dt <= timestamp <= end_dt:
                    price = float(item['price'])
                    min_price = float(item.get('minPrice', price))
                    max_price = float(item.get('maxPrice', price))
                    
                    price_data.append({
                        'timestamp': timestamp,
                        'mark_open': price,
                        'mark_high': max_price,
                        'mark_low': min_price,
                        'mark_close': price,
                        'index_open': price,  # GMX uses oracle price as index
                        'index_high': max_price,
                        'index_low': min_price,
                        'index_close': price,
                        'volume': 0,  # Volume not available in this query
                        'open_interest': 0  # OI not available in this query
                    })
            
            if not price_data:
                logger.warning(f"No price data found for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(price_data)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            
            # Resample to hourly data for consistency
            df_hourly = df.resample('1H').agg({
                'mark_open': 'first',
                'mark_high': 'max',
                'mark_low': 'min',
                'mark_close': 'last',
                'index_open': 'first',
                'index_high': 'max',
                'index_low': 'min',
                'index_close': 'last',
                'volume': 'sum',
                'open_interest': 'last'
            }).dropna()
            
            logger.info(f"Fetched {len(df_hourly)} price records for {symbol}")
            return df_hourly
            
        except Exception as e:
            logger.error(f"Failed to fetch GMX price data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def test_connection(self) -> bool:
        """Test connection to GMX subgraph."""
        try:
            query = "{ _meta { block { number } } }"
            response = requests.post(
                self.subgraph_url,
                json={'query': query},
                timeout=10
            )
            response.raise_for_status()
            logger.info("GMX subgraph connection successful")
            return True
        except Exception as e:
            logger.error(f"GMX subgraph connection failed: {str(e)}")
            return False

def test_gmx_collector():
    """Test the GMX collector with a single symbol."""
    collector = GmxCollector()
    
    # Test connection
    if not collector.test_connection():
        return False
    
    # Test markets list
    markets = collector.fetch_markets_list()
    if not markets:
        logger.error("No markets found")
        return False
    
    logger.info(f"Found {len(markets)} markets")
    for market in markets[:5]:  # Show first 5
        logger.info(f"  {market['symbol']} - {market['id']}")
    
    # Test data collection for BTC
    btc_markets = [m for m in markets if 'BTC' in m['symbol'].upper()]
    if btc_markets:
        symbol = btc_markets[0]['symbol']
        logger.info(f"Testing data collection for {symbol}")
        
        # Collect last 7 days of data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        df = collector.collect_symbol_data(
            symbol, 
            start_time.isoformat(), 
            end_time.isoformat()
        )
        
        if df is not None and not df.empty:
            logger.info(f"Successfully collected {len(df)} records")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            logger.info(f"Sample data:\n{df.head()}")
            return True
        else:
            logger.error("No data collected")
            return False
    else:
        logger.error("No BTC market found for testing")
        return False

if __name__ == "__main__":
    test_gmx_collector()
