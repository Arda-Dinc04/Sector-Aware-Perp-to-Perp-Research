"""
Extended Binance data collector for multi-venue research.
Collects funding rates, mark prices, index prices, and market data from Binance.
This serves as our working baseline while we fix other venue APIs.
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

class BinanceExtendedCollector(BaseDataCollector):
    """Extended Binance data collector for multi-venue research."""
    
    def __init__(self, data_dir: str = "raw_data"):
        super().__init__("binance", data_dir)
        self.base_url = "https://fapi.binance.com"
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    def get_taker_fees(self) -> float:
        """Binance futures taker fees: 0.04% = 4 bps"""
        return 4.0
    
    def get_maker_fees(self) -> float:
        """Binance futures maker fees: 0.02% = 2 bps"""
        return 2.0
    
    def get_borrow_fees(self) -> float:
        """Binance doesn't have borrow fees for perps"""
        return 0.0
    
    def fetch_markets_list(self) -> List[Dict[str, Any]]:
        """Fetch list of available markets from Binance."""
        try:
            url = f"{self.base_url}/fapi/v1/exchangeInfo"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            markets = []
            
            for symbol_info in data.get('symbols', []):
                if symbol_info.get('status') == 'TRADING':
                    markets.append({
                        'symbol': symbol_info['symbol'],
                        'base_asset': symbol_info['baseAsset'],
                        'quote_asset': symbol_info['quoteAsset'],
                        'tick_size': float(symbol_info.get('filters', [{}])[0].get('tickSize', 0)),
                        'step_size': float(symbol_info.get('filters', [{}])[1].get('stepSize', 0)),
                        'min_qty': float(symbol_info.get('filters', [{}])[1].get('minQty', 0))
                    })
            
            logger.info(f"Found {len(markets)} active markets on Binance")
            return markets
            
        except Exception as e:
            logger.error(f"Failed to fetch Binance markets: {str(e)}")
            return []
    
    def fetch_funding_rates(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch funding rates for a symbol from Binance."""
        try:
            # Convert times to timestamps
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            
            url = f"{self.base_url}/fapi/v1/fundingRate"
            start_ts = int(start_dt.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)
            params = {
                'symbol': symbol,
                'startTime': start_ts,
                'endTime': end_ts,
                'limit': 1000
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            funding_data = []
            
            for item in data:
                timestamp = pd.to_datetime(int(item['fundingTime']), unit='ms')
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
            logger.error(f"Failed to fetch Binance funding rates for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_price_data(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch price data for a symbol from Binance."""
        try:
            # Convert times to timestamps
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            
            # Get 1-hour klines
            url = f"{self.base_url}/fapi/v1/klines"
            start_ts = int(start_dt.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)
            params = {
                'symbol': symbol,
                'interval': '1h',
                'startTime': start_ts,
                'endTime': end_ts,
                'limit': 1000
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            price_data = []
            
            for kline in data:
                timestamp = pd.to_datetime(int(kline[0]), unit='ms')
                if start_dt <= timestamp <= end_dt:
                    price_data.append({
                        'timestamp': timestamp,
                        'mark_open': float(kline[1]),
                        'mark_high': float(kline[2]),
                        'mark_low': float(kline[3]),
                        'mark_close': float(kline[4]),
                        'volume': float(kline[5]),
                        'open_interest': float(kline[8]) if len(kline) > 8 else 0
                    })
            
            if not price_data:
                logger.warning(f"No price data found for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(price_data)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            
            # Get index prices (spot proxy)
            index_df = self.fetch_index_prices(symbol, start_time, end_time)
            if not index_df.empty:
                df = pd.merge(df, index_df, left_index=True, right_index=True, how='left')
            else:
                # Use mark prices as index proxy
                df['index_open'] = df['mark_open']
                df['index_high'] = df['mark_high']
                df['index_low'] = df['mark_low']
                df['index_close'] = df['mark_close']
            
            logger.info(f"Fetched {len(df)} price records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch Binance price data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_index_prices(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch index prices for a symbol from Binance."""
        try:
            # Convert symbol format (BTCUSDT -> BTC)
            base_asset = symbol.replace('USDT', '').replace('BUSD', '')
            
            # Get spot price history (simplified - using current price as proxy)
            url = f"{self.base_url.replace('fapi', 'api')}/v3/klines"
            start_ts = int(pd.to_datetime(start_time).timestamp() * 1000)
            end_ts = int(pd.to_datetime(end_time).timestamp() * 1000)
            params = {
                'symbol': f"{base_asset}USDT",
                'interval': '1h',
                'startTime': start_ts,
                'endTime': end_ts,
                'limit': 1000
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            index_data = []
            
            for kline in data:
                timestamp = pd.to_datetime(int(kline[0]), unit='ms')
                index_data.append({
                    'timestamp': timestamp,
                    'index_open': float(kline[1]),
                    'index_high': float(kline[2]),
                    'index_low': float(kline[3]),
                    'index_close': float(kline[4])
                })
            
            if not index_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(index_data)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch index prices for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def test_connection(self) -> bool:
        """Test connection to Binance API."""
        try:
            url = f"{self.base_url}/fapi/v1/ping"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            logger.info("Binance API connection successful")
            return True
        except Exception as e:
            logger.error(f"Binance API connection failed: {str(e)}")
            return False

def test_binance_extended_collector():
    """Test the extended Binance collector."""
    collector = BinanceExtendedCollector()
    
    # Test connection
    if not collector.test_connection():
        return False
    
    # Test markets list
    markets = collector.fetch_markets_list()
    if not markets:
        logger.error("No markets found")
        return False
    
    logger.info(f"Found {len(markets)} markets")
    
    # Test data collection for BTCUSDT
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
    test_binance_extended_collector()
