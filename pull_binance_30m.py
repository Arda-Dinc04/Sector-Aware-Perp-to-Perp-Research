#!/usr/bin/env python3
"""
Binance USDⓈ-M Futures Data Collector (30-minute resolution, ~3 years)

Pulls funding rates, mark/index prices, volume, and open interest data
from Binance Futures API and saves as Parquet files.

Dependencies: requests, pandas, pyarrow
"""

import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Generator
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BinanceFuturesCollector:
    """Binance USDⓈ-M Futures data collector with rate limiting and pagination."""

    def __init__(self, base_url: Optional[str] = None):
        env_base_url = os.getenv("BINANCE_FAPI_BASE")
        self.base_url = base_url or env_base_url or "https://fapi.binance.com"
        self.testnet_base_url = os.getenv("BINANCE_FAPI_TESTNET")
        self.allow_testnet_fallback = os.getenv("BINANCE_ALLOW_TESTNET_FALLBACK", "false").lower() in {"1", "true", "yes"}
        self._using_testnet = False
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BinanceFuturesCollector/1.0'
        })

        # Rate limiting settings (conservative)
        self.max_weight_per_minute = 1500  # Target <2400 limit
        self.request_weights = {
            'fundingRate': 1,
            'markPriceKlines': 5,  # Weight depends on limit
            'indexPriceKlines': 5,
            'klines': 5,
            'openInterestHist': 1,
            'exchangeInfo': 10
        }
        self.last_request_time = 0
        self.current_weight = 0
        self.weight_reset_time = time.time()

        # Sector mapping (will filter available symbols)
        self.sector_map = {
            "Majors": ["BTCUSDT", "ETHUSDT"],
            "DeFi": ["UNIUSDT", "AAVEUSDT", "CRVUSDT", "COMPUSDT"],
            "Layer2_Infra": ["ARBUSDT", "OPUSDT", "MATICUSDT"],
            "AI_Gaming": ["IMXUSDT", "RNDRUSDT", "AGIXUSDT", "GALAUSDT"],
            "Memes": ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT"]
        }

        # Data directory
        self.data_dir = Path("data/binance_30m")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit_guard(self, endpoint: str, response: requests.Response) -> None:
        """Monitor rate limits and sleep if necessary."""
        current_time = time.time()

        # Reset weight counter every minute
        if current_time - self.weight_reset_time >= 60:
            self.current_weight = 0
            self.weight_reset_time = current_time

        # Add weight for this request
        weight = self.request_weights.get(endpoint, 1)
        self.current_weight += weight

        # Check response headers for actual usage
        used_weight = response.headers.get('X-MBX-USED-WEIGHT')
        if used_weight:
            logger.debug(f"API weight used: {used_weight}")

        # If we're approaching limits, sleep
        if self.current_weight >= self.max_weight_per_minute * 0.8:
            sleep_time = 60 - (current_time - self.weight_reset_time)
            if sleep_time > 0:
                logger.info(f"Rate limit protection: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self.current_weight = 0
                self.weight_reset_time = time.time()

        # Standard inter-request delay
        time.sleep(0.1)

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request with rate limiting and error handling."""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            self._rate_limit_guard(endpoint.split('/')[-1], response)

            if response.status_code == 451:
                message = (
                    "Binance returned HTTP 451 (unavailable for legal reasons). "
                    "This usually means your IP is geoblocked."
                )
                if self.allow_testnet_fallback and self.testnet_base_url and not self._using_testnet:
                    logger.warning("%s Switching to testnet base URL %s", message, self.testnet_base_url)
                    self.base_url = self.testnet_base_url
                    self._using_testnet = True
                    return self._make_request(endpoint, params)

                raise RuntimeError(
                    message +
                    " Set BINANCE_FAPI_BASE to an allowed endpoint or route through a compliant region."
                )

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited, sleeping {retry_after}s")
                time.sleep(retry_after)
                return self._make_request(endpoint, params)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get_available_symbols(self) -> Dict[str, List[str]]:
        """Get available symbols from exchange info, filtered by sector map."""
        logger.info("Fetching available symbols from /fapi/v1/exchangeInfo")

        data = self._make_request("/fapi/v1/exchangeInfo", {})
        available_symbols = {s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING'}

        filtered_sectors = {}
        for sector, symbols in self.sector_map.items():
            available = [s for s in symbols if s in available_symbols]
            if available:
                filtered_sectors[sector] = available
                logger.info(f"{sector}: {available}")
            else:
                logger.warning(f"{sector}: no symbols available")

        return filtered_sectors

    def paginate_time_range(self, start_time: int, end_time: int,
                           step_days: int) -> Generator[Tuple[int, int], None, None]:
        """Generate time chunks for pagination."""
        current_start = start_time
        step_ms = step_days * 24 * 60 * 60 * 1000

        while current_start < end_time:
            current_end = min(current_start + step_ms, end_time)
            yield current_start, current_end
            current_start = current_end

    @staticmethod
    def interval_to_milliseconds(interval: str) -> int:
        """Convert Binance interval string (e.g. '30m', '1h') to milliseconds."""
        if not interval:
            raise ValueError("Interval string must be non-empty")

        unit = interval[-1]
        try:
            value = int(interval[:-1])
        except ValueError as exc:
            raise ValueError(f"Invalid interval value in '{interval}'") from exc

        multipliers = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000,
            'M': 30 * 24 * 60 * 60 * 1000  # Binance treats '1M' as 1 month
        }

        if unit not in multipliers:
            raise ValueError(f"Unsupported interval unit '{unit}' in '{interval}'")

        return value * multipliers[unit]

    def get_funding_rate_history(self, symbol: str, start_time: int,
                                end_time: int) -> pd.DataFrame:
        """Pull funding rate history with pagination."""
        logger.info(f"Pulling funding rate history for {symbol}")

        all_data = []
        for chunk_start, chunk_end in self.paginate_time_range(start_time, end_time, 120):
            params = {
                'symbol': symbol,
                'startTime': chunk_start,
                'endTime': chunk_end,
                'limit': 1000
            }

            data = self._make_request("/fapi/v1/fundingRate", params)
            if data:
                all_data.extend(data)
                logger.debug(f"Fetched {len(data)} funding rate records")

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
        df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
        df['markPrice'] = pd.to_numeric(df['markPrice'], errors='coerce')

        return df.sort_values('fundingTime').reset_index(drop=True)

    def get_klines(self, endpoint: str, symbol: str, interval: str,
                   start_time: int, end_time: int, **kwargs) -> pd.DataFrame:
        """Generic klines puller for mark price, index price, and regular klines."""
        logger.info(f"Pulling {endpoint} data for {symbol}")

        limit = 1000  # Binance hard limit for kline endpoints
        interval_ms = self.interval_to_milliseconds(interval)

        request_key = 'pair' if 'pair' in kwargs else 'symbol'
        request_value = kwargs.get('pair', symbol)

        all_data: List[List] = []
        next_start = start_time

        while next_start < end_time:
            params = {
                request_key: request_value,
                'interval': interval,
                'startTime': next_start,
                'endTime': end_time,
                'limit': limit
            }

            data = self._make_request(endpoint, params)

            if not data:
                break

            all_data.extend(data)
            logger.debug(f"Fetched {len(data)} kline records starting at {next_start}")

            if len(data) < limit:
                break

            last_open_time = int(data[-1][0])
            next_start = last_open_time + interval_ms

            if next_start <= last_open_time:
                logger.warning("Detected non-increasing kline timestamps; stopping pagination early")
                break

        if not all_data:
            return pd.DataFrame()

        # Standard klines format: [openTime, open, high, low, close, volume, closeTime, ...]
        df = pd.DataFrame(all_data, columns=[
            'openTime', 'open', 'high', 'low', 'close', 'volume',
            'closeTime', 'quoteVolume', 'count', 'takerBuyVolume', 'takerBuyQuoteVolume', 'ignore'
        ])

        df['openTime'] = pd.to_datetime(df['openTime'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume', 'quoteVolume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.sort_values('openTime').reset_index(drop=True)

    def get_open_interest_history(self, symbol: str, start_time: int,
                                 end_time: int) -> pd.DataFrame:
        """Pull open interest history (limited to ~30 days)."""
        logger.info(f"Pulling open interest history for {symbol}")

        # Limit to last 30 days due to API constraints
        thirty_days_ago = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
        actual_start = max(start_time, thirty_days_ago)

        if actual_start >= end_time:
            logger.warning(f"No OI data available for {symbol} in requested range")
            return pd.DataFrame()

        limit = 500
        interval_ms = self.interval_to_milliseconds('30m')

        all_data: List[Dict] = []
        next_start = actual_start

        while next_start < end_time:
            params = {
                'symbol': symbol,
                'period': '30m',
                'startTime': next_start,
                'endTime': end_time,
                'limit': limit
            }

            try:
                data = self._make_request("/futures/data/openInterestHist", params)
            except Exception as e:
                logger.warning(f"OI request failed for {symbol}: {e}")
                break

            if not data:
                break

            all_data.extend(data)
            logger.debug(f"Fetched {len(data)} OI records starting at {next_start}")

            if len(data) < limit:
                break

            last_timestamp = int(data[-1]['timestamp'])
            next_start = last_timestamp + interval_ms

            if next_start <= last_timestamp:
                logger.warning("Detected non-increasing OI timestamps; stopping pagination early")
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce')
        df['sumOpenInterestValue'] = pd.to_numeric(df['sumOpenInterestValue'], errors='coerce')

        return df.sort_values('timestamp').reset_index(drop=True)

    def resample_funding_to_30m(self, funding_df: pd.DataFrame,
                               timestamp_grid: pd.DatetimeIndex) -> pd.DataFrame:
        """Resample 8h funding rates to 30m intervals using provided timestamp grid."""
        if funding_df.empty or len(timestamp_grid) == 0:
            return pd.DataFrame()

        # Forward fill funding rates
        funding_df = funding_df.set_index('fundingTime')

        result_data = []
        for ts in timestamp_grid:
            # Find the most recent funding rate
            prior_funding = funding_df[funding_df.index <= ts]
            if not prior_funding.empty:
                latest_funding = prior_funding.iloc[-1]
                funding_rate_interval = latest_funding['fundingRate']
                # Convert 8h rate to 30m accrual
                funding_rate_30m = funding_rate_interval * (30 / (8 * 60))  # 30min / 480min
            else:
                funding_rate_interval = np.nan
                funding_rate_30m = np.nan

            result_data.append({
                'timestamp': ts,
                'funding_rate_interval': funding_rate_interval,
                'funding_rate_30m': funding_rate_30m
            })

        return pd.DataFrame(result_data)

    def collect_symbol_data(self, symbol: str, sector: str) -> pd.DataFrame:
        """Collect all data for a single symbol and merge into 30m panel."""
        logger.info(f"Collecting data for {symbol} ({sector})")

        # Time range: ~3 years
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=3*365)).timestamp() * 1000)

        # 1. Funding rate history
        funding_df = self.get_funding_rate_history(symbol, start_time, end_time)

        # 2. Mark price klines
        mark_df = self.get_klines("/fapi/v1/markPriceKlines", symbol, "30m",
                                 start_time, end_time)

        # 3. Index price klines
        index_df = self.get_klines("/fapi/v1/indexPriceKlines", symbol, "30m",
                                  start_time, end_time, pair=symbol)

        # 4. Regular klines (for volume)
        klines_df = self.get_klines("/fapi/v1/klines", symbol, "30m",
                                   start_time, end_time)

        # 5. Open interest history
        oi_df = self.get_open_interest_history(symbol, start_time, end_time)

        # 6. Merge all data on timestamp (create timestamp grid from klines)
        return self.merge_data_to_panel(symbol, sector, funding_df, mark_df,
                                       index_df, klines_df, oi_df)

    def merge_data_to_panel(self, symbol: str, sector: str, funding_df: pd.DataFrame,
                           mark_df: pd.DataFrame, index_df: pd.DataFrame,
                           klines_df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all data sources into a single 30m panel."""

        # Use klines as the primary timestamp grid (most complete for 30m intervals)
        if klines_df.empty:
            logger.warning(f"No klines data for {symbol}, skipping")
            return pd.DataFrame()

        # Start with klines data and rename columns
        klines_df = klines_df.rename(columns={
            'openTime': 'timestamp',
            'open': 'price_open',
            'high': 'price_high',
            'low': 'price_low',
            'close': 'price_close'
        })

        panel = klines_df[['timestamp', 'price_open', 'price_high', 'price_low',
                          'price_close', 'volume', 'quoteVolume']].copy()

        # Add metadata
        panel['exchange'] = 'Binance'
        panel['symbol'] = symbol
        panel['sector'] = sector

        # Resample funding rates to match klines timestamps
        if not funding_df.empty:
            funding_30m = self.resample_funding_to_30m(funding_df, panel['timestamp'])
            panel = panel.merge(funding_30m[['timestamp', 'funding_rate_interval', 'funding_rate_30m']],
                               on='timestamp', how='left')
        else:
            panel['funding_rate_interval'] = np.nan
            panel['funding_rate_30m'] = np.nan

        # Merge mark price data
        if not mark_df.empty:
            mark_df = mark_df.rename(columns={
                'openTime': 'timestamp',
                'open': 'mark_open',
                'high': 'mark_high',
                'low': 'mark_low',
                'close': 'mark_close'
            })
            panel = panel.merge(mark_df[['timestamp', 'mark_open', 'mark_high',
                                        'mark_low', 'mark_close']],
                               on='timestamp', how='left')
        else:
            for col in ['mark_open', 'mark_high', 'mark_low', 'mark_close']:
                panel[col] = np.nan

        # Merge index price data
        if not index_df.empty:
            index_df = index_df.rename(columns={
                'openTime': 'timestamp',
                'open': 'index_open',
                'high': 'index_high',
                'low': 'index_low',
                'close': 'index_close'
            })
            panel = panel.merge(index_df[['timestamp', 'index_open', 'index_high',
                                         'index_low', 'index_close']],
                               on='timestamp', how='left')
        else:
            for col in ['index_open', 'index_high', 'index_low', 'index_close']:
                panel[col] = np.nan

        # Merge open interest data
        if not oi_df.empty:
            oi_df = oi_df.rename(columns={'sumOpenInterestValue': 'open_interest'})
            panel = panel.merge(oi_df[['timestamp', 'open_interest']],
                               on='timestamp', how='left')
        else:
            panel['open_interest'] = np.nan

        # Reorder columns to match specification
        column_order = [
            'timestamp', 'exchange', 'symbol', 'sector',
            'funding_rate_interval', 'funding_rate_30m',
            'mark_open', 'mark_high', 'mark_low', 'mark_close',
            'index_open', 'index_high', 'index_low', 'index_close',
            'price_open', 'price_high', 'price_low', 'price_close',
            'volume', 'quoteVolume', 'open_interest'
        ]

        return panel[column_order].sort_values('timestamp').reset_index(drop=True)

    def save_to_parquet(self, df: pd.DataFrame, symbol: str) -> None:
        """Save dataframe to Parquet file."""
        if df.empty:
            logger.warning(f"No data to save for {symbol}")
            return

        filepath = self.data_dir / f"{symbol}.parquet"
        df.to_parquet(filepath, engine='pyarrow', compression='snappy')
        logger.info(f"Saved {len(df)} records for {symbol} to {filepath}")

    def run_collection(self) -> None:
        """Main collection workflow."""
        logger.info("Starting Binance Futures data collection")

        # Get available symbols
        available_symbols = self.get_available_symbols()

        # Collect data for each symbol
        for sector, symbols in available_symbols.items():
            for symbol in symbols:
                try:
                    panel_data = self.collect_symbol_data(symbol, sector)
                    self.save_to_parquet(panel_data, symbol)
                except Exception as e:
                    logger.error(f"Failed to collect data for {symbol}: {e}")
                    continue

        logger.info("Data collection completed")


if __name__ == "__main__":
    collector = BinanceFuturesCollector()
    collector.run_collection()
