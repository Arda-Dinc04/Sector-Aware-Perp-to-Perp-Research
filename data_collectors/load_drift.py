"""
Drift data collector for perpetual futures data.
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

class DriftCollector(BaseDataCollector):
    """Drift data collector."""
    
    def __init__(self, data_dir: str = "raw_data"):
        super().__init__("drift", data_dir)
        # Drift uses their own API
        self.base_url = "https://api.drift.trade/api"
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    def get_taker_fees(self) -> float:
        """Drift taker fees: 0.1% = 10 bps"""
        return 10.0
    
    def get_maker_fees(self) -> float:
        """Drift maker fees: 0.02% = 2 bps"""
        return 2.0
    
    def get_borrow_fees(self) -> float:
        """Drift borrow fees: 0.01% = 1 bps per day"""
        return 1.0
    
    def fetch_markets_list(self) -> List[Dict[str, Any]]:
        """Fetch list of available markets from Drift."""
        try:
            url = f"{self.base_url}/markets"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            markets = []
            
            for market in data.get('markets', []):
                if market.get('status') == 'active':
                    markets.append({
                        'symbol': market['name'],
                        'id': market['marketIndex'],
                        'base_asset': market.get('baseAsset', ''),
                        'quote_asset': market.get('quoteAsset', 'USDC'),
                        'tick_size': float(market.get('tickSize', 0)),
                        'step_size': float(market.get('stepSize', 0)),
                        'min_order_size': float(market.get('minOrderSize', 0))
                    })
            
            logger.info(f"Found {len(markets)} active markets on Drift")
            return markets
            
        except Exception as e:
            logger.error(f"Failed to fetch Drift markets: {str(e)}")
            return []
    
    def fetch_funding_rates(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch funding rates for a symbol from Drift."""
        try:
            # Convert times to timestamps
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            
            # Drift funding rates endpoint
            url = f"{self.base_url}/funding-rates"
            params = {
                'market': symbol,
                'from': int(start_dt.timestamp()),
                'to': int(end_dt.timestamp())
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            funding_data = []
            
            for item in data.get('fundingRates', []):
                timestamp = pd.to_datetime(item['timestamp'])
                if start_dt <= timestamp <= end_dt:
                    funding_data.append({
                        'timestamp': timestamp,
                        'funding_rate': float(item['fundingRate'])
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
            logger.error(f"Failed to fetch Drift funding rates for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_price_data(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch price data for a symbol from Drift."""
        try:
            # Convert times to timestamps
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            
            # Drift price data endpoint
            url = f"{self.base_url}/price-history"
            params = {
                'market': symbol,
                'resolution': '1h',  # 1-hour candles
                'from': int(start_dt.timestamp()),
                'to': int(end_dt.timestamp())
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
                timestamp = pd.to_datetime(candle['timestamp'])
                if start_dt <= timestamp <= end_dt:
                    price_data.append({
                        'timestamp': timestamp,
                        'mark_open': float(candle['open']),
                        'mark_high': float(candle['high']),
                        'mark_low': float(candle['low']),
                        'mark_close': float(candle['close']),
                        'index_open': float(candle.get('indexOpen', candle['open'])),
                        'index_high': float(candle.get('indexHigh', candle['high'])),
                        'index_low': float(candle.get('indexLow', candle['low'])),
                        'index_close': float(candle.get('indexClose', candle['close'])),
                        'volume': float(candle.get('volume', 0)),
                        'open_interest': float(candle.get('openInterest', 0))
                    })
            
            df = pd.DataFrame(price_data)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            
            logger.info(f"Fetched {len(df)} price records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch Drift price data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def test_connection(self) -> bool:
        """Test connection to Drift API."""
        try:
            url = f"{self.base_url}/markets"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            logger.info("Drift API connection successful")
            return True
        except Exception as e:
            logger.error(f"Drift API connection failed: {str(e)}")
            return False

def test_drift_collector():
    """Test the Drift collector with a single symbol."""
    collector = DriftCollector()
    
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
    
    # Test data collection for BTC-PERP
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
    test_drift_collector()
