#!/usr/bin/env python3
"""
Sector-Aware Perp-to-Perp Research - Baseline Backtester

Implements two baseline strategies:
1. Buy & Hold (spot) - equal weight portfolio
2. Naive Hedged Carry - market-neutral funding capture

Author: Research Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class BacktesterConfig:
    """Configuration class for backtester parameters."""
    
    def __init__(self):
        # Data parameters
        self.use_index_as_spot = True
        self.funding_window = "8h"  # Binance funding period
        self.data_dir = "data/binance_30m"
        
        # Position sizing
        self.notional_usd = 10000  # Per symbol
        self.leverage = 1.0
        
        # Fees (in basis points)
        self.fees = {
            'spot_taker_bps': 10,    # 0.10%
            'perp_taker_bps': 4.5,   # 0.045%
            'perp_maker_bps': 0,     # Assume taker for baseline
            'borrow_bps_per_day': 0  # No borrow costs for now
        }
        
        # Strategy parameters
        self.epsilon_bps = 2.0  # Buffer above fees (2 bps as agreed)
        self.basis_stop_bps = 100  # Stop loss on basis drift
        self.bh_rebalance = None  # None = never, "monthly" = monthly
        
        # Trading configuration
        self.maker_fraction = 0.5      # 50% maker / 50% taker assumption
        self.hysteresis_in_bps = 12    # Enter threshold (bps)
        self.hysteresis_out_bps = 6    # Exit threshold (bps)
        
        # Symbol selection
        self.symbols = "all"  # "all" or list of specific symbols
        self.majors_only = ["BTCUSDT", "ETHUSDT"]
        
        # Performance metrics
        self.rolling_sharpe_windows = [30, 60]  # Days
        self.risk_free_rate = 0.02  # 2% annual risk-free rate

class Backtester:
    """Main backtester class for baseline strategies."""
    
    def __init__(self, config: BacktesterConfig = None):
        self.config = config or BacktesterConfig()
        self.data = {}
        self.results = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess all parquet files."""
        print("Loading data...")
        
        import os
        import glob
        
        parquet_files = glob.glob(f"{self.config.data_dir}/*.parquet")
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.config.data_dir}")
        
        # Filter out future dates
        today = pd.Timestamp.now(tz='UTC')
        
        for file_path in parquet_files:
            symbol = os.path.basename(file_path).replace('.parquet', '')
            
            # Load data
            df = pd.read_parquet(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Filter out future dates
            df = df[df['timestamp'] <= today]
            
            if len(df) == 0:
                print(f"Warning: No valid data for {symbol}")
                continue
                
            # Set timestamp as index for easier resampling
            df = df.set_index('timestamp')
            
            # Ensure we have the required columns
            required_cols = ['funding_rate_30m', 'mark_open', 'mark_high', 'mark_low', 'mark_close',
                           'index_open', 'index_high', 'index_low', 'index_close',
                           'price_open', 'price_high', 'price_low', 'price_close', 'volume']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: {symbol} missing columns: {missing_cols}")
                continue
            
            self.data[symbol] = df
            print(f"Loaded {symbol}: {len(df)} records from {df.index[0]} to {df.index[-1]}")
        
        print(f"Successfully loaded {len(self.data)} symbols")
        return self.data
    
    def resample_to_funding_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 30m data to 8h funding periods."""
        # Binance funding occurs at 00:00, 08:00, 16:00 UTC
        # Align to these exact times
        
        # Resample to 8h periods aligned to UTC funding times
        agg_dict = {
            'funding_rate_30m': 'sum',  # Sum of 30m accruals = 8h rate
            'mark_open': 'first',
            'mark_high': 'max', 
            'mark_low': 'min',
            'mark_close': 'last',
            'index_open': 'first',
            'index_high': 'max',
            'index_low': 'min', 
            'index_close': 'last',
            'price_open': 'first',
            'price_high': 'max',
            'price_low': 'min',
            'price_close': 'last',
            'volume': 'sum'
        }
        
        # Add optional columns if they exist
        if 'quoteVolume' in df.columns:
            agg_dict['quoteVolume'] = 'sum'
        if 'open_interest' in df.columns:
            agg_dict['open_interest'] = 'last'
        
        resampled = df.resample('8H', origin='epoch', offset='0H').agg(agg_dict)
        
        # Drop any periods with all NaN values
        resampled = resampled.dropna(how='all')
        
        return resampled
    
    def calculate_fees(self, notional: float, is_spot: bool = False) -> float:
        """Calculate trading fees for a given notional with maker/taker fraction."""
        if is_spot:
            # Spot trades are always taker
            return notional * (self.config.fees['spot_taker_bps'] / 10000)
        else:
            # Perp trades: mix of maker and taker
            maker_rate = self.config.fees['perp_maker_bps'] / 10000
            taker_rate = self.config.fees['perp_taker_bps'] / 10000
            effective_rate = (self.config.maker_fraction * maker_rate + 
                            (1 - self.config.maker_fraction) * taker_rate)
            return notional * effective_rate
    
    def buy_and_hold_baseline(self, symbols: List[str] = None) -> Dict:
        """Implement Buy & Hold baseline strategy."""
        print("Running Buy & Hold baseline...")
        
        if symbols is None:
            symbols = list(self.data.keys())
        
        # Initialize portfolio
        portfolio_value = 0
        positions = {}
        fees_paid = 0
        
        # Get common time range across all symbols
        min_time = max([self.data[symbol].index[0] for symbol in symbols])
        max_time = min([self.data[symbol].index[-1] for symbol in symbols])
        
        print(f"Trading period: {min_time} to {max_time}")
        
        # Resample all data to 8h periods
        resampled_data = {}
        for symbol in symbols:
            resampled_data[symbol] = self.resample_to_funding_periods(self.data[symbol])
        
        # Find common time index
        common_times = set(resampled_data[symbols[0]].index)
        for symbol in symbols[1:]:
            common_times = common_times.intersection(set(resampled_data[symbol].index))
        
        common_times = sorted(list(common_times))
        common_times = [t for t in common_times if min_time <= t <= max_time]
        
        if not common_times:
            raise ValueError("No common time periods found across symbols")
        
        # Initial allocation (equal weight)
        notional_per_symbol = self.config.notional_usd
        total_notional = notional_per_symbol * len(symbols)
        
        for symbol in symbols:
            # Use spot prices (index_close) for Buy & Hold
            spot_price = resampled_data[symbol].loc[common_times[0], 'index_close']
            if pd.isna(spot_price):
                print(f"Warning: No spot price for {symbol} at {common_times[0]}")
                continue
                
            # Calculate position size
            position_size = notional_per_symbol / spot_price
            
            # Calculate fees for initial purchase
            spot_fees = self.calculate_fees(notional_per_symbol, is_spot=True)
            fees_paid += spot_fees
            
            positions[symbol] = {
                'size': position_size,
                'entry_price': spot_price,
                'entry_time': common_times[0]
            }
        
        # Track portfolio value over time
        portfolio_values = []
        timestamps = []
        
        for timestamp in common_times:
            current_value = 0
            
            for symbol, pos in positions.items():
                if timestamp in resampled_data[symbol].index:
                    spot_price = resampled_data[symbol].loc[timestamp, 'index_close']
                    if not pd.isna(spot_price):
                        current_value += pos['size'] * spot_price
            
            portfolio_values.append(current_value)
            timestamps.append(timestamp)
        
        # Calculate returns
        portfolio_series = pd.Series(portfolio_values, index=timestamps)
        returns = portfolio_series.pct_change().dropna()
        
        # Add exit fees for final liquidation
        exit_fees = 0
        for symbol, pos in positions.items():
            if pos['size'] > 0:  # Only if we have a position
                final_spot_price = resampled_data[symbol].loc[common_times[-1], 'index_close']
                if not pd.isna(final_spot_price):
                    exit_fees += self.calculate_fees(pos['size'] * final_spot_price, is_spot=True)
        
        # Adjust final portfolio value for exit fees
        final_portfolio_value = portfolio_values[-1] - exit_fees
        total_fees_paid = fees_paid + exit_fees
        
        # Calculate final metrics (8h periods: 3 per day = 1095 per year)
        PERIODS_PER_YEAR = 365 * 3  # 8h periods
        total_return = final_portfolio_value / total_notional - 1
        mean_ret = returns.mean()
        vol_ret = returns.std()
        annualized_return = (1 + total_return) ** (PERIODS_PER_YEAR / len(returns)) - 1
        annualized_vol = vol_ret * np.sqrt(PERIODS_PER_YEAR)
        sharpe = (annualized_return - self.config.risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        results = {
            'strategy': 'Buy & Hold',
            'symbols': symbols,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': annualized_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'fees_paid': total_fees_paid,
            'portfolio_values': portfolio_series,
            'returns': returns,
            'drawdown': drawdown,
            'positions': positions
        }
        
        print(f"Buy & Hold Results:")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Annualized Return: {annualized_return:.2%}")
        print(f"  Volatility: {annualized_vol:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Fees Paid: ${total_fees_paid:.2f}")
        
        return results
    
    def naive_hedged_carry_baseline(self, symbols: List[str] = None) -> Dict:
        """Implement Naive Hedged Carry baseline strategy (perp-to-perp)."""
        print("Running Naive Hedged Carry baseline (perp-to-perp)...")
        
        if symbols is None:
            symbols = list(self.data.keys())
        
        # Initialize tracking
        portfolio_values = []
        timestamps = []
        total_fees = 0
        total_funding_pnl = 0
        total_basis_pnl = 0
        
        # Get common time range and resample data
        resampled_data = {}
        for symbol in symbols:
            resampled_data[symbol] = self.resample_to_funding_periods(self.data[symbol])
        
        # Find common time index
        common_times = set(resampled_data[symbols[0]].index)
        for symbol in symbols[1:]:
            common_times = common_times.intersection(set(resampled_data[symbol].index))
        
        common_times = sorted(list(common_times))
        
        if not common_times:
            raise ValueError("No common time periods found across symbols")
        
        # Initialize positions for each symbol (perp-to-perp: long and short perp positions)
        positions = {symbol: {'long_perp_size': 0, 'short_perp_size': 0, 'entry_basis': 0, 'prev_basis': 0, 'in_position': False} for symbol in symbols}
        
        # Start with initial capital
        initial_capital = self.config.notional_usd * len(symbols)
        current_capital = initial_capital
        
        for i, timestamp in enumerate(common_times):
            period_pnl = 0
            period_fees = 0
            period_funding = 0
            period_basis = 0
            
            for symbol in symbols:
                if timestamp not in resampled_data[symbol].index:
                    continue
                    
                data = resampled_data[symbol].loc[timestamp]
                
                # Get prices
                spot_price = data['index_close']
                perp_price = data['price_close']
                funding_rate = data['funding_rate_30m']
                
                if pd.isna(spot_price) or pd.isna(perp_price) or pd.isna(funding_rate):
                    continue
                
                # Calculate threshold for perp-to-perp trading
                # Perp-to-perp: 2 legs × effective perp rate + epsilon
                effective_perp_rate = (self.config.maker_fraction * self.config.fees['perp_maker_bps'] + 
                                     (1 - self.config.maker_fraction) * self.config.fees['perp_taker_bps'])
                total_fees_bps = 2 * effective_perp_rate  # Both legs are perp trades
                threshold = (total_fees_bps + self.config.epsilon_bps) / 10000
                
                # Hysteresis logic: different thresholds for entry and exit
                entry_threshold = self.config.hysteresis_in_bps / 10000
                exit_threshold = self.config.hysteresis_out_bps / 10000
                
                # Determine position based on funding rate with hysteresis
                if not positions[symbol]['in_position'] and abs(funding_rate) > entry_threshold:
                    # Calculate leveraged notional
                    leveraged_notional = self.config.notional_usd * self.config.leverage
                    
                    # Determine direction
                    if funding_rate > 0:
                        # Short perp (funding payer), long perp (funding receiver)
                        target_long_perp = leveraged_notional / perp_price
                        target_short_perp = -leveraged_notional / perp_price
                    else:
                        # Long perp (funding receiver), short perp (funding payer)
                        target_long_perp = leveraged_notional / perp_price
                        target_short_perp = -leveraged_notional / perp_price
                    
                    # Calculate position changes
                    long_change = target_long_perp - positions[symbol]['long_perp_size']
                    short_change = target_short_perp - positions[symbol]['short_perp_size']
                    
                    # Calculate fees for trades (both are perp trades)
                    if abs(long_change) > 1e-8:
                        long_fees = self.calculate_fees(abs(long_change) * perp_price, is_spot=False)
                        period_fees += long_fees
                    
                    if abs(short_change) > 1e-8:
                        short_fees = self.calculate_fees(abs(short_change) * perp_price, is_spot=False)
                        period_fees += short_fees
                    
                    # Update positions
                    positions[symbol]['long_perp_size'] = target_long_perp
                    positions[symbol]['short_perp_size'] = target_short_perp
                    positions[symbol]['entry_basis'] = 0  # No basis in perp-to-perp
                    positions[symbol]['prev_basis'] = 0
                    positions[symbol]['in_position'] = True
                elif positions[symbol]['in_position'] and abs(funding_rate) < exit_threshold:
                    # Exit position when signal drops below exit threshold (hysteresis)
                    if positions[symbol]['long_perp_size'] != 0 or positions[symbol]['short_perp_size'] != 0:
                        # Calculate fees for closing trades
                        if abs(positions[symbol]['long_perp_size']) > 1e-8:
                            long_fees = self.calculate_fees(abs(positions[symbol]['long_perp_size']) * perp_price, is_spot=False)
                            period_fees += long_fees
                        
                        if abs(positions[symbol]['short_perp_size']) > 1e-8:
                            short_fees = self.calculate_fees(abs(positions[symbol]['short_perp_size']) * perp_price, is_spot=False)
                            period_fees += short_fees
                        
                        # Close positions
                        positions[symbol] = {'long_perp_size': 0, 'short_perp_size': 0, 'entry_basis': 0, 'prev_basis': 0, 'in_position': False}
                
                # Calculate PnL for this period
                if positions[symbol]['long_perp_size'] != 0 or positions[symbol]['short_perp_size'] != 0:
                    # Funding PnL for long positions
                    if positions[symbol]['long_perp_size'] != 0:
                        funding_pnl = positions[symbol]['long_perp_size'] * perp_price * funding_rate
                        period_funding += funding_pnl
                    
                    # Funding PnL for short positions (short pays funding)
                    if positions[symbol]['short_perp_size'] != 0:
                        funding_pnl = abs(positions[symbol]['short_perp_size']) * perp_price * funding_rate
                        period_funding += funding_pnl
            
            # Update capital
            period_pnl = period_funding + period_basis - period_fees
            current_capital += period_pnl
            total_fees += period_fees
            total_funding_pnl += period_funding
            total_basis_pnl += period_basis
            
            portfolio_values.append(current_capital)
            timestamps.append(timestamp)
        
        # Calculate returns
        portfolio_series = pd.Series(portfolio_values, index=timestamps)
        returns = portfolio_series.pct_change().dropna()
        
        # Calculate metrics (8h periods: 3 per day = 1095 per year)
        PERIODS_PER_YEAR = 365 * 3  # 8h periods
        total_return = portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1
        mean_ret = returns.mean()
        vol_ret = returns.std()
        annualized_return = (1 + total_return) ** (PERIODS_PER_YEAR / len(returns)) - 1
        annualized_vol = vol_ret * np.sqrt(PERIODS_PER_YEAR)
        sharpe = (annualized_return - self.config.risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        results = {
            'strategy': 'Naive Hedged Carry',
            'symbols': symbols,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': annualized_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'fees_paid': total_fees,
            'funding_pnl': total_funding_pnl,
            'basis_pnl': total_basis_pnl,
            'portfolio_values': portfolio_series,
            'returns': returns,
            'drawdown': drawdown,
            'positions': positions
        }
        
        print(f"Naive Hedged Carry Results:")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Annualized Return: {annualized_return:.2%}")
        print(f"  Volatility: {annualized_vol:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Fees Paid: ${total_fees:.2f}")
        print(f"  Funding PnL: ${total_funding_pnl:.2f}")
        print(f"  Basis PnL: ${total_basis_pnl:.2f}")
        
        return results

    def run_baselines(self):
        """Run both baseline strategies."""
        print("=" * 60)
        print("SECTOR-AWARE PERP-TO-PERP RESEARCH - BASELINE BACKTEST")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Determine symbols to use
        if self.config.symbols == "all":
            symbols = list(self.data.keys())
        else:
            symbols = self.config.symbols
        
        print(f"\nRunning baselines on {len(symbols)} symbols: {symbols}")
        
        # Run Buy & Hold baseline
        print("\n" + "=" * 40)
        bh_results = self.buy_and_hold_baseline(symbols)
        self.results['buy_hold'] = bh_results
        
        # Run Naive Hedged Carry baseline
        print("\n" + "=" * 40)
        carry_results = self.naive_hedged_carry_baseline(symbols)
        self.results['hedged_carry'] = carry_results
        
        
        # Run Majors-only comparison
        print("\n" + "=" * 40)
        print("MAJORS-ONLY COMPARISON (BTC, ETH)")
        print("=" * 40)
        
        majors_symbols = [s for s in self.config.majors_only if s in symbols]
        if majors_symbols:
            bh_majors = self.buy_and_hold_baseline(majors_symbols)
            carry_majors = self.naive_hedged_carry_baseline(majors_symbols)
            
            self.results['buy_hold_majors'] = bh_majors
            self.results['hedged_carry_majors'] = carry_majors
        
        print("\n" + "=" * 60)
        print("BASELINE BACKTEST COMPLETED")
        print("=" * 60)
        
        return self.results

if __name__ == "__main__":
    # Create and run backtester
    config = BacktesterConfig()
    backtester = Backtester(config)
    results = backtester.run_baselines()
