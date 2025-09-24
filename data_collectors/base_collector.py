"""
Base data collector for multi-venue perp data collection.
Provides common utilities for data normalization and 8h resampling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseDataCollector(ABC):
    """Base class for venue-specific data collectors."""
    
    def __init__(self, venue_name: str, data_dir: str = "raw_data"):
        self.venue_name = venue_name
        self.data_dir = os.path.join(data_dir, venue_name)
        self.symbols_map = self._load_symbols_map()
        
        # Create venue directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _load_symbols_map(self) -> pd.DataFrame:
        """Load the symbols mapping configuration."""
        symbols_path = os.path.join("config", "symbols_map.csv")
        if os.path.exists(symbols_path):
            return pd.read_csv(symbols_path)
        else:
            logger.warning(f"Symbols map not found at {symbols_path}")
            return pd.DataFrame()
    
    def normalize_8h(self, df: pd.DataFrame, funding_col: str, 
                    price_cols: str, idx_cols: str) -> pd.DataFrame:
        """
        Normalize data to 8h buckets aligned to UTC funding windows.
        
        Args:
            df: DataFrame with UTC timestamp index
            funding_col: Column name for funding rate
            price_cols: Prefix for price columns (e.g., 'mark' or 'last')
            idx_cols: Prefix for index columns (e.g., 'index')
        
        Returns:
            DataFrame resampled to 8h buckets
        """
        if df.empty:
            return df
            
        # Ensure timestamp is timezone-aware UTC
        if not df.index.tz:
            df.index = df.index.tz_localize('UTC')
        elif df.index.tz != 'UTC':
            df.index = df.index.tz_convert('UTC')
        
        # Define aggregation rules
        agg_rules = {
            funding_col: 'sum',  # Sum funding accruals over 8h
            'volume': 'sum',     # Sum volume over 8h
            'open_interest': 'last'  # Last OI in the period
        }
        
        # Add price aggregations
        for prefix in [price_cols, idx_cols]:
            if f'{prefix}_open' in df.columns:
                agg_rules[f'{prefix}_open'] = 'first'
            if f'{prefix}_high' in df.columns:
                agg_rules[f'{prefix}_high'] = 'max'
            if f'{prefix}_low' in df.columns:
                agg_rules[f'{prefix}_low'] = 'min'
            if f'{prefix}_close' in df.columns:
                agg_rules[f'{prefix}_close'] = 'last'
        
        # Resample to 8h buckets aligned to UTC funding windows
        resampled = df.resample('8H', origin='epoch', offset='0H').agg(agg_rules)
        
        # Drop periods with all NaN values
        resampled = resampled.dropna(how='all')
        
        # Add 30m funding rate (prorated from 8h)
        if funding_col in resampled.columns:
            resampled['funding_rate_30m'] = resampled[funding_col] * (30 / 480)  # 30min / 8h
        
        return resampled
    
    def standardize_schema(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Standardize DataFrame to canonical schema.
        
        Args:
            df: Raw venue data
            symbol: Symbol name
            
        Returns:
            Standardized DataFrame with canonical columns
        """
        # Get asset and sector from mapping
        venue_symbols = self.symbols_map[self.symbols_map['venue'] == self.venue_name]
        symbol_info = venue_symbols[venue_symbols['symbol'] == symbol]
        
        if symbol_info.empty:
            logger.warning(f"Symbol {symbol} not found in mapping for venue {self.venue_name}")
            asset = symbol
            sector = "UNKNOWN"
        else:
            asset = symbol_info.iloc[0]['asset']
            sector = symbol_info.iloc[0]['sector']
        
        # Create standardized DataFrame
        standardized = pd.DataFrame(index=df.index)
        standardized['venue'] = self.venue_name
        standardized['symbol'] = symbol
        standardized['asset'] = asset
        standardized['sector'] = sector
        
        # Map funding rate (ensure positive = longs pay)
        if 'funding_rate' in df.columns:
            standardized['funding_rate'] = df['funding_rate']
        elif 'funding_rate_8h' in df.columns:
            standardized['funding_rate'] = df['funding_rate_8h']
        else:
            standardized['funding_rate'] = 0.0
        
        # Map 30m funding rate
        if 'funding_rate_30m' in df.columns:
            standardized['funding_rate_30m'] = df['funding_rate_30m']
        else:
            standardized['funding_rate_30m'] = standardized['funding_rate'] * (30 / 480)
        
        # Map price data
        price_mapping = {
            'mark_open': 'mark_o',
            'mark_high': 'mark_h', 
            'mark_low': 'mark_l',
            'mark_close': 'mark_c',
            'index_open': 'index_o',
            'index_high': 'index_h',
            'index_low': 'index_l', 
            'index_close': 'index_c',
            'last_open': 'last_o',
            'last_high': 'last_h',
            'last_low': 'last_l',
            'last_close': 'last_c'
        }
        
        for old_col, new_col in price_mapping.items():
            if old_col in df.columns:
                standardized[new_col] = df[old_col]
            else:
                standardized[new_col] = np.nan
        
        # Map other fields
        if 'open_interest' in df.columns:
            standardized['open_interest'] = df['open_interest']
        else:
            standardized['open_interest'] = np.nan
            
        if 'volume' in df.columns:
            standardized['volume'] = df['volume']
        else:
            standardized['volume'] = np.nan
        
        # Add fee information (venue-specific defaults)
        standardized['fees_taker_bps_perp'] = self.get_taker_fees()
        standardized['fees_maker_bps_perp'] = self.get_maker_fees()
        standardized['borrow_bps_per_day'] = self.get_borrow_fees()
        
        return standardized
    
    @abstractmethod
    def get_taker_fees(self) -> float:
        """Get venue-specific taker fees in bps."""
        pass
    
    @abstractmethod
    def get_maker_fees(self) -> float:
        """Get venue-specific maker fees in bps."""
        pass
    
    @abstractmethod
    def get_borrow_fees(self) -> float:
        """Get venue-specific borrow fees in bps per day."""
        pass
    
    @abstractmethod
    def fetch_markets_list(self) -> List[Dict[str, Any]]:
        """Fetch list of available markets from venue."""
        pass
    
    @abstractmethod
    def fetch_funding_rates(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch funding rates for a symbol."""
        pass
    
    @abstractmethod
    def fetch_price_data(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch price data for a symbol."""
        pass
    
    def collect_symbol_data(self, symbol: str, start_time: str, end_time: str) -> Optional[pd.DataFrame]:
        """
        Collect all data for a single symbol.
        
        Args:
            symbol: Symbol to collect
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            
        Returns:
            Standardized DataFrame or None if collection failed
        """
        try:
            logger.info(f"Collecting {symbol} data from {self.venue_name}...")
            
            # Fetch funding rates
            funding_df = self.fetch_funding_rates(symbol, start_time, end_time)
            
            # Fetch price data
            price_df = self.fetch_price_data(symbol, start_time, end_time)
            
            # Merge data
            if funding_df.empty and price_df.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Merge on timestamp
            if not funding_df.empty and not price_df.empty:
                df = pd.merge(funding_df, price_df, left_index=True, right_index=True, how='outer')
            elif not funding_df.empty:
                df = funding_df
            else:
                df = price_df
            
            # Normalize to 8h buckets
            df_normalized = self.normalize_8h(df, 'funding_rate', 'mark', 'index')
            
            # Standardize schema
            df_standardized = self.standardize_schema(df_normalized, symbol)
            
            # Save raw data
            output_path = os.path.join(self.data_dir, f"{symbol}.parquet")
            df_standardized.to_parquet(output_path)
            
            logger.info(f"Successfully collected {len(df_standardized)} records for {symbol}")
            return df_standardized
            
        except Exception as e:
            logger.error(f"Failed to collect {symbol} from {self.venue_name}: {str(e)}")
            return None
    
    def collect_all_symbols(self, start_time: str, end_time: str) -> Dict[str, pd.DataFrame]:
        """
        Collect data for all available symbols.
        
        Args:
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        markets = self.fetch_markets_list()
        results = {}
        
        for market in markets:
            symbol = market['symbol']
            df = self.collect_symbol_data(symbol, start_time, end_time)
            if df is not None:
                results[symbol] = df
        
        logger.info(f"Collected data for {len(results)} symbols from {self.venue_name}")
        return results
