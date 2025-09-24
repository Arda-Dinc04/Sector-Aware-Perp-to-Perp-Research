#!/usr/bin/env python3
"""
Visualization module for baseline backtest results.

Creates comprehensive charts for PnL, Sharpe ratios, drawdowns, and performance metrics.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')

class BaselineVisualizer:
    """Visualization class for baseline backtest results."""
    
    def __init__(self, results: Dict, config=None):
        self.results = results
        self.config = config
        
    def plot_cumulative_pnl(self, figsize=(15, 8), save_path=None):
        """Plot cumulative PnL for all strategies."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Baseline Strategies - Cumulative PnL', fontsize=16, fontweight='bold')
        
        # All symbols
        ax1 = axes[0, 0]
        if 'buy_hold' in self.results:
            bh_data = self.results['buy_hold']['portfolio_values']
            ax1.plot(bh_data.index, bh_data.values, label='Buy & Hold', linewidth=2)
        
        if 'hedged_carry' in self.results:
            carry_data = self.results['hedged_carry']['portfolio_values']
            ax1.plot(carry_data.index, carry_data.values, label='Hedged Carry', linewidth=2)
        
        ax1.set_title('All Symbols')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Majors only
        ax2 = axes[0, 1]
        if 'buy_hold_majors' in self.results:
            bh_majors = self.results['buy_hold_majors']['portfolio_values']
            ax2.plot(bh_majors.index, bh_majors.values, label='Buy & Hold (Majors)', linewidth=2)
        
        if 'hedged_carry_majors' in self.results:
            carry_majors = self.results['hedged_carry_majors']['portfolio_values']
            ax2.plot(carry_majors.index, carry_majors.values, label='Hedged Carry (Majors)', linewidth=2)
        
        ax2.set_title('Majors Only (BTC, ETH)')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Normalized returns (starting at 100)
        ax3 = axes[1, 0]
        if 'buy_hold' in self.results:
            bh_norm = (self.results['buy_hold']['portfolio_values'] / 
                      self.results['buy_hold']['portfolio_values'].iloc[0] * 100)
            ax3.plot(bh_norm.index, bh_norm.values, label='Buy & Hold', linewidth=2)
        
        if 'hedged_carry' in self.results:
            carry_norm = (self.results['hedged_carry']['portfolio_values'] / 
                         self.results['hedged_carry']['portfolio_values'].iloc[0] * 100)
            ax3.plot(carry_norm.index, carry_norm.values, label='Hedged Carry', linewidth=2)
        
        ax3.set_title('Normalized Returns (Base = 100)')
        ax3.set_ylabel('Index Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Majors normalized
        ax4 = axes[1, 1]
        if 'buy_hold_majors' in self.results:
            bh_majors_norm = (self.results['buy_hold_majors']['portfolio_values'] / 
                             self.results['buy_hold_majors']['portfolio_values'].iloc[0] * 100)
            ax4.plot(bh_majors_norm.index, bh_majors_norm.values, label='Buy & Hold (Majors)', linewidth=2)
        
        if 'hedged_carry_majors' in self.results:
            carry_majors_norm = (self.results['hedged_carry_majors']['portfolio_values'] / 
                               self.results['hedged_carry_majors']['portfolio_values'].iloc[0] * 100)
            ax4.plot(carry_majors_norm.index, carry_majors_norm.values, label='Hedged Carry (Majors)', linewidth=2)
        
        ax4.set_title('Majors Normalized Returns (Base = 100)')
        ax4.set_ylabel('Index Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Don't show interactive plots by default
        # plt.show()
    
    def plot_rolling_sharpe(self, windows=[30, 60], figsize=(15, 6), save_path=None):
        """Plot rolling Sharpe ratios."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Rolling Sharpe Ratios', fontsize=16, fontweight='bold')
        
        for i, window in enumerate(windows):
            ax = axes[i]
            
            if 'buy_hold' in self.results:
                returns = self.results['buy_hold']['returns']
                rolling_mean = returns.rolling(window=window*3).mean()
                rolling_std = returns.rolling(window=window*3).std()
                rolling_sharpe = rolling_mean / rolling_std * np.sqrt(365*3)
                rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)  # Handle division by zero
                ax.plot(rolling_sharpe.index, rolling_sharpe.values, label='Buy & Hold', linewidth=2)
            
            if 'hedged_carry' in self.results:
                returns = self.results['hedged_carry']['returns']
                rolling_mean = returns.rolling(window=window*3).mean()
                rolling_std = returns.rolling(window=window*3).std()
                rolling_sharpe = rolling_mean / rolling_std * np.sqrt(365*3)
                rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)  # Handle division by zero
                ax.plot(rolling_sharpe.index, rolling_sharpe.values, label='Hedged Carry', linewidth=2)
            
            ax.set_title(f'{window}-Day Rolling Sharpe')
            ax.set_ylabel('Sharpe Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Don't show interactive plots by default
        # plt.show()
    
    def plot_drawdowns(self, figsize=(15, 8), save_path=None):
        """Plot drawdown charts."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Drawdown Analysis', fontsize=16, fontweight='bold')
        
        # All symbols drawdowns
        ax1 = axes[0, 0]
        if 'buy_hold' in self.results:
            drawdown = self.results['buy_hold']['drawdown'] * 100
            ax1.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, label='Buy & Hold')
        
        if 'hedged_carry' in self.results:
            drawdown = self.results['hedged_carry']['drawdown'] * 100
            ax1.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, label='Hedged Carry')
        
        ax1.set_title('All Symbols - Drawdowns')
        ax1.set_ylabel('Drawdown (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Majors drawdowns
        ax2 = axes[0, 1]
        if 'buy_hold_majors' in self.results:
            drawdown = self.results['buy_hold_majors']['drawdown'] * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, label='Buy & Hold (Majors)')
        
        if 'hedged_carry_majors' in self.results:
            drawdown = self.results['hedged_carry_majors']['drawdown'] * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, label='Hedged Carry (Majors)')
        
        ax2.set_title('Majors Only - Drawdowns')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Underwater plots (cumulative max)
        ax3 = axes[1, 0]
        if 'buy_hold' in self.results:
            portfolio_values = self.results['buy_hold']['portfolio_values']
            running_max = portfolio_values.expanding().max()
            underwater = (portfolio_values - running_max) / running_max * 100
            ax3.fill_between(underwater.index, underwater.values, 0, alpha=0.7, label='Buy & Hold')
        
        if 'hedged_carry' in self.results:
            portfolio_values = self.results['hedged_carry']['portfolio_values']
            running_max = portfolio_values.expanding().max()
            underwater = (portfolio_values - running_max) / running_max * 100
            ax3.fill_between(underwater.index, underwater.values, 0, alpha=0.7, label='Hedged Carry')
        
        ax3.set_title('All Symbols - Underwater Plot')
        ax3.set_ylabel('Underwater (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Majors underwater
        ax4 = axes[1, 1]
        if 'buy_hold_majors' in self.results:
            portfolio_values = self.results['buy_hold_majors']['portfolio_values']
            running_max = portfolio_values.expanding().max()
            underwater = (portfolio_values - running_max) / running_max * 100
            ax4.fill_between(underwater.index, underwater.values, 0, alpha=0.7, label='Buy & Hold (Majors)')
        
        if 'hedged_carry_majors' in self.results:
            portfolio_values = self.results['hedged_carry_majors']['portfolio_values']
            running_max = portfolio_values.expanding().max()
            underwater = (portfolio_values - running_max) / running_max * 100
            ax4.fill_between(underwater.index, underwater.values, 0, alpha=0.7, label='Hedged Carry (Majors)')
        
        ax4.set_title('Majors Only - Underwater Plot')
        ax4.set_ylabel('Underwater (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Don't show interactive plots by default
        # plt.show()
    
    def plot_returns_distribution(self, figsize=(15, 6), save_path=None):
        """Plot returns distribution and statistics."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Returns Distribution Analysis', fontsize=16, fontweight='bold')
        
        # All symbols
        ax1 = axes[0]
        if 'buy_hold' in self.results:
            returns = self.results['buy_hold']['returns'] * 100
            ax1.hist(returns.dropna(), bins=50, alpha=0.7, label='Buy & Hold', density=True)
        
        if 'hedged_carry' in self.results:
            returns = self.results['hedged_carry']['returns'] * 100
            ax1.hist(returns.dropna(), bins=50, alpha=0.7, label='Hedged Carry', density=True)
        
        ax1.set_title('All Symbols - Returns Distribution')
        ax1.set_xlabel('Returns (%)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Majors only
        ax2 = axes[1]
        if 'buy_hold_majors' in self.results:
            returns = self.results['buy_hold_majors']['returns'] * 100
            ax2.hist(returns.dropna(), bins=50, alpha=0.7, label='Buy & Hold (Majors)', density=True)
        
        if 'hedged_carry_majors' in self.results:
            returns = self.results['hedged_carry_majors']['returns'] * 100
            ax2.hist(returns.dropna(), bins=50, alpha=0.7, label='Hedged Carry (Majors)', density=True)
        
        ax2.set_title('Majors Only - Returns Distribution')
        ax2.set_xlabel('Returns (%)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Don't show interactive plots by default
        # plt.show()
    
    def create_performance_table(self, save_path=None):
        """Create a comprehensive performance metrics table."""
        strategies = []
        
        # Collect all strategy results
        for key, result in self.results.items():
            if isinstance(result, dict) and 'strategy' in result:
                strategies.append({
                    'Strategy': result['strategy'],
                    'Symbols': key.split('_')[-1] if '_' in key else 'All',
                    'Total Return (%)': f"{result['total_return']:.2f}",
                    'Annualized Return (%)': f"{result['annualized_return']:.2f}",
                    'Volatility (%)': f"{result['volatility']:.2f}",
                    'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
                    'Max Drawdown (%)': f"{result['max_drawdown']:.2f}",
                    'Fees Paid ($)': f"{result['fees_paid']:.2f}",
                })
                
                # Add carry-specific metrics
                if 'funding_pnl' in result:
                    strategies[-1]['Funding PnL ($)'] = f"{result['funding_pnl']:.2f}"
                    if 'basis_pnl' in result:
                        strategies[-1]['Basis PnL ($)'] = f"{result['basis_pnl']:.2f}"
                    else:
                        strategies[-1]['Basis PnL ($)'] = "N/A"
        
        # Create DataFrame
        df = pd.DataFrame(strategies)
        
        # Display table
        print("\n" + "=" * 100)
        print("PERFORMANCE METRICS SUMMARY")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\nPerformance table saved to: {save_path}")
        
        return df
    
    def create_all_plots(self, save_dir=None):
        """Create all visualization plots."""
        print("Creating comprehensive visualization suite...")
        
        # Create plots
        self.plot_cumulative_pnl(save_path=f"{save_dir}/cumulative_pnl.png" if save_dir else None)
        self.plot_rolling_sharpe(save_path=f"{save_dir}/rolling_sharpe.png" if save_dir else None)
        self.plot_drawdowns(save_path=f"{save_dir}/drawdowns.png" if save_dir else None)
        self.plot_returns_distribution(save_path=f"{save_dir}/returns_distribution.png" if save_dir else None)
        
        # Create performance table
        self.create_performance_table(save_path=f"{save_dir}/performance_metrics.csv" if save_dir else None)
        
        print("All visualizations completed!")

if __name__ == "__main__":
    # Example usage
    print("Visualizer module loaded. Use with backtest results.")
