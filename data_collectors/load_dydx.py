"""
dYdX data collector for perpetual futures data.
Collects funding rates, mark prices, index prices, and market data.
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

class DyDxCollector(BaseDataCollector):
    """dYdX data collector."""
    
    def __init__(self, data_dir: str = "raw_data"):
        super().__init__("dydx", data_dir)
        self.base_url = "https://api.dydx.exchange/v3"
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    def get_taker_fees(self) -> float:
        """dYdX taker fees: 0.05% = 5 bps"""
        return 5.0
    
    def get_maker_fees(self) -> float:
        """dYdX maker fees: 0.02% = 2 bps"""
        return 2.0
    
    def get_borrow_fees(self) -> float:
        """dYdX doesn't have borrow fees for perps"""
        return 0.0
    
    def fetch_markets_list(self) -> List[Dict[str, Any]]:
        """Fetch list of available markets from dYdX."""
        try:
            url = f"{self.base_url}/markets"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            markets = []
            
            for market_id, market_data in data['markets'].items():
                if market_data.get('status') == 'ONLINE':
                    markets.append({
                        'symbol': market_data['market'],
                        'id': market_id,
                        'tick_size': float(market_data.get('tickSize', 0)),
                        'step_size': float(market_data.get('stepSize', 0)),
                        'min_order_size': float(market_data.get('minOrderSize', 0)),
                        'funding_interval': market_data.get('fundingInterval', '8H')
                    })
            
            logger.info(f"Found {len(markets)} active markets on dYdX")
            return markets
            
        except Exception as e:
            logger.error(f"Failed to fetch dYdX markets: {str(e)}")
            return []
    
    def fetch_funding_rates(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch funding rates for a symbol from dYdX."""
        try:
            # Convert times to timestamps
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            
            # dYdX funding rates are published every 8 hours
            # We'll fetch the funding rate history
            url = f"{self.base_url}/historical-funding"
            params = {
                'market': symbol,
                'effectiveBeforeOrAt': end_dt.isoformat() + 'Z'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            funding_data = []
            
            for item in data.get('historicalFunding', []):
                timestamp = pd.to_datetime(item['effectiveAt'])
                if start_dt <= timestamp <= end_dt:
                    funding_data.append({
                        'timestamp': timestamp,
                        'funding_rate': float(item['rate'])
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
            logger.error(f"Failed to fetch dYdX funding rates for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_price_data(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch price data for a symbol from dYdX."""
        try:
            # Convert times to timestamps
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            
            # dYdX provides candlestick data
            url = f"{self.base_url}/candles/{symbol}"
            params = {
                'resolution': '1HOUR',  # 1-hour candles
                'fromISO': start_dt.isoformat() + 'Z',
                'toISO': end_dt.isoformat() + 'Z'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            candles = data.get('candles', [])
            
            if not candles:
                logger.warning(f"No price data found for {symbol}")
                return pd.DataFrame()
            
            price_data = []
            for candle in candles:
                timestamp = pd.to_datetime(candle['startedAt'])
                if start_dt <= timestamp <= end_dt:
                    price_data.append({
                        'timestamp': timestamp,
                        'mark_open': float(candle['open']),
                        'mark_high': float(candle['high']),
                        'mark_low': float(candle['low']),
                        'mark_close': float(candle['close']),
                        'volume': float(candle['usdVolume']),
                        'open_interest': float(candle.get('openInterest', 0))
                    })
            
            df = pd.DataFrame(price_data)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            
            # For dYdX, we need to fetch index prices separately
            # For now, we'll use mark prices as proxy (this should be improved)
            df['index_open'] = df['mark_open']
            df['index_high'] = df['mark_high']
            df['index_low'] = df['mark_low']
            df['index_close'] = df['mark_close']
            
            logger.info(f"Fetched {len(df)} price records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch dYdX price data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def test_connection(self) -> bool:
        """Test connection to dYdX API."""
        try:
            url = f"{self.base_url}/markets"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            logger.info("dYdX API connection successful")
            return True
        except Exception as e:
            logger.error(f"dYdX API connection failed: {str(e)}")
            return False

def test_dydx_collector():
    """Test the dYdX collector with a single symbol."""
    collector = DyDxCollector()
    
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
    
    # Test data collection for BTC-USD
    btc_markets = [m for m in markets if 'BTC' in m['symbol']]
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
    test_dydx_collector()
