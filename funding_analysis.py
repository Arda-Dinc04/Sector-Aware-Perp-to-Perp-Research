#!/usr/bin/env python3
"""
Comprehensive Funding Rate Analysis
Analyzes funding rate patterns, signal coverage, and strategy performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtester import Backtester, BacktesterConfig
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FundingAnalyzer:
    """Comprehensive funding rate and strategy analysis."""
    
    def __init__(self, config: BacktesterConfig = None):
        self.config = config or BacktesterConfig()
        self.data = {}
        self.resampled_data = {}
        
    def load_and_prepare_data(self):
        """Load and resample data for analysis."""
        print("Loading and preparing data for analysis...")
        
        backtester = Backtester(self.config)
        backtester.load_data()
        self.data = backtester.data
        
        # Resample all data to 8h funding periods
        for symbol, df in self.data.items():
            self.resampled_data[symbol] = backtester.resample_to_funding_periods(df)
        
        print(f"Loaded {len(self.data)} symbols for analysis")
        
    def calculate_signal_coverage(self, threshold_bps: float = 16.5) -> Dict:
        """Calculate signal coverage for different thresholds."""
        print(f"\n{'='*60}")
        print(f"SIGNAL COVERAGE ANALYSIS (Threshold: {threshold_bps} bps)")
        print(f"{'='*60}")
        
        threshold = threshold_bps / 10000  # Convert to decimal
        
        coverage_results = {}
        
        for symbol, df in self.resampled_data.items():
            if 'funding_rate_30m' not in df.columns:
                continue
                
            # Calculate absolute funding rate
            abs_funding = df['funding_rate_30m'].abs()
            
            # Count active windows
            active_windows = (abs_funding > threshold).sum()
            total_windows = len(abs_funding.dropna())
            
            if total_windows > 0:
                coverage_pct = (active_windows / total_windows) * 100
            else:
                coverage_pct = 0
                
            coverage_results[symbol] = {
                'active_windows': active_windows,
                'total_windows': total_windows,
                'coverage_pct': coverage_pct,
                'avg_funding_when_active': abs_funding[abs_funding > threshold].mean() if active_windows > 0 else 0,
                'median_funding_when_active': abs_funding[abs_funding > threshold].median() if active_windows > 0 else 0,
                'max_funding': abs_funding.max(),
                'min_funding': abs_funding.min()
            }
            
            print(f"{symbol:10s}: {coverage_pct:6.2f}% coverage ({active_windows:4d}/{total_windows:4d} windows)")
        
        return coverage_results
    
    def analyze_funding_magnitude(self) -> Dict:
        """Analyze funding rate magnitude patterns."""
        print(f"\n{'='*60}")
        print("FUNDING MAGNITUDE ANALYSIS")
        print(f"{'='*60}")
        
        magnitude_results = {}
        
        for symbol, df in self.resampled_data.items():
            if 'funding_rate_30m' not in df.columns:
                continue
                
            funding_rates = df['funding_rate_30m'].dropna()
            
            if len(funding_rates) == 0:
                continue
                
            magnitude_results[symbol] = {
                'mean_abs_funding': funding_rates.abs().mean(),
                'median_abs_funding': funding_rates.abs().median(),
                'std_abs_funding': funding_rates.abs().std(),
                'max_abs_funding': funding_rates.abs().max(),
                'min_abs_funding': funding_rates.abs().min(),
                'positive_funding_pct': (funding_rates > 0).mean() * 100,
                'negative_funding_pct': (funding_rates < 0).mean() * 100,
                'zero_funding_pct': (funding_rates == 0).mean() * 100
            }
            
            print(f"{symbol:10s}: Mean={funding_rates.abs().mean():.6f}, "
                  f"Median={funding_rates.abs().median():.6f}, "
                  f"Max={funding_rates.abs().max():.6f}")
        
        return magnitude_results
    
    def analyze_strategy_performance(self) -> Dict:
        """Analyze detailed strategy performance metrics."""
        print(f"\n{'='*60}")
        print("STRATEGY PERFORMANCE ANALYSIS")
        print(f"{'='*60}")
        
        # Run backtester to get detailed results
        backtester = Backtester(self.config)
        backtester.data = self.data
        
        strategy_results = {}
        
        for symbol in self.data.keys():
            print(f"\nAnalyzing {symbol}...")
            
            try:
                # Run Buy & Hold
                bh_result = backtester.buy_and_hold_baseline([symbol])
                
                # Run Hedged Carry
                carry_result = backtester.naive_hedged_carry_baseline([symbol])
                
                # Calculate additional metrics
                if 'portfolio_values' in carry_result:
                    portfolio_series = carry_result['portfolio_values']
                    returns = portfolio_series.pct_change().dropna()
                    
                    # Trade analysis
                    position_changes = (returns != 0).sum()
                    holding_periods = len(returns) / max(position_changes, 1)
                    
                    # PnL decomposition
                    total_pnl = portfolio_series.iloc[-1] - portfolio_series.iloc[0]
                    funding_pnl = carry_result.get('funding_pnl', 0)
                    fees_paid = carry_result.get('fees_paid', 0)
                    basis_pnl = carry_result.get('basis_pnl', 0)
                    
                    strategy_results[symbol] = {
                        'bh_total_return': bh_result['total_return'],
                        'bh_annualized_return': bh_result['annualized_return'],
                        'bh_sharpe': bh_result['sharpe_ratio'],
                        'carry_total_return': carry_result['total_return'],
                        'carry_annualized_return': carry_result['annualized_return'],
                        'carry_sharpe': carry_result['sharpe_ratio'],
                        'total_pnl': total_pnl,
                        'funding_pnl': funding_pnl,
                        'fees_paid': fees_paid,
                        'basis_pnl': basis_pnl,
                        'trade_count': position_changes,
                        'avg_holding_periods': holding_periods,
                        'funding_pnl_pct': (funding_pnl / total_pnl * 100) if total_pnl != 0 else 0,
                        'fees_pnl_pct': (fees_paid / total_pnl * 100) if total_pnl != 0 else 0,
                        'basis_pnl_pct': (basis_pnl / total_pnl * 100) if total_pnl != 0 else 0
                    }
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                strategy_results[symbol] = None
        
        return strategy_results
    
    def analyze_majors_vs_non_majors(self) -> Dict:
        """Compare majors vs non-majors funding patterns."""
        print(f"\n{'='*60}")
        print("MAJORS VS NON-MAJORS ANALYSIS")
        print(f"{'='*60}")
        
        majors = ['BTCUSDT', 'ETHUSDT']
        non_majors = [s for s in self.data.keys() if s not in majors]
        
        threshold = 16.5 / 10000  # 16.5 bps threshold
        
        results = {
            'majors': {'symbols': majors, 'coverage': {}, 'magnitude': {}},
            'non_majors': {'symbols': non_majors, 'coverage': {}, 'magnitude': {}}
        }
        
        for category, symbols in [('majors', majors), ('non_majors', non_majors)]:
            if not symbols:
                continue
                
            print(f"\n{category.upper()} ANALYSIS:")
            
            coverage_pcts = []
            magnitudes = []
            
            for symbol in symbols:
                if symbol not in self.resampled_data:
                    continue
                    
                df = self.resampled_data[symbol]
                if 'funding_rate_30m' not in df.columns:
                    continue
                    
                abs_funding = df['funding_rate_30m'].abs()
                active_windows = (abs_funding > threshold).sum()
                total_windows = len(abs_funding.dropna())
                
                if total_windows > 0:
                    coverage_pct = (active_windows / total_windows) * 100
                    avg_magnitude = abs_funding.mean()
                    
                    coverage_pcts.append(coverage_pct)
                    magnitudes.append(avg_magnitude)
                    
                    results[category]['coverage'][symbol] = coverage_pct
                    results[category]['magnitude'][symbol] = avg_magnitude
                    
                    print(f"  {symbol:10s}: {coverage_pct:6.2f}% coverage, {avg_magnitude:.6f} avg magnitude")
            
            if coverage_pcts:
                results[category]['avg_coverage'] = np.mean(coverage_pcts)
                results[category]['avg_magnitude'] = np.mean(magnitudes)
                print(f"  Average: {np.mean(coverage_pcts):6.2f}% coverage, {np.mean(magnitudes):.6f} avg magnitude")
        
        return results
    
    def create_funding_visualizations(self):
        """Create visualizations for funding rate analysis."""
        print(f"\n{'='*60}")
        print("CREATING FUNDING ANALYSIS VISUALIZATIONS")
        print(f"{'='*60}")
        
        # 1. Funding rate distribution by symbol
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Funding Rate Analysis', fontsize=16)
        
        # Plot 1: Funding rate coverage by symbol
        symbols = list(self.resampled_data.keys())
        coverage_data = []
        
        threshold = 16.5 / 10000
        for symbol in symbols:
            if symbol in self.resampled_data and 'funding_rate_30m' in self.resampled_data[symbol].columns:
                abs_funding = self.resampled_data[symbol]['funding_rate_30m'].abs()
                active_windows = (abs_funding > threshold).sum()
                total_windows = len(abs_funding.dropna())
                coverage_pct = (active_windows / total_windows) * 100 if total_windows > 0 else 0
                coverage_data.append(coverage_pct)
            else:
                coverage_data.append(0)
        
        axes[0, 0].bar(symbols, coverage_data, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Signal Coverage by Symbol (16.5 bps threshold)')
        axes[0, 0].set_ylabel('Coverage %')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Average funding magnitude by symbol
        magnitude_data = []
        for symbol in symbols:
            if symbol in self.resampled_data and 'funding_rate_30m' in self.resampled_data[symbol].columns:
                avg_magnitude = self.resampled_data[symbol]['funding_rate_30m'].abs().mean()
                magnitude_data.append(avg_magnitude)
            else:
                magnitude_data.append(0)
        
        axes[0, 1].bar(symbols, magnitude_data, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Average Funding Magnitude by Symbol')
        axes[0, 1].set_ylabel('Average |Funding Rate|')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Funding rate time series (sample)
        sample_symbols = symbols[:3]  # Show first 3 symbols
        for i, symbol in enumerate(sample_symbols):
            if symbol in self.resampled_data and 'funding_rate_30m' in self.resampled_data[symbol].columns:
                df = self.resampled_data[symbol]
                axes[1, 0].plot(df.index, df['funding_rate_30m'], label=symbol, alpha=0.7)
        
        axes[1, 0].set_title('Funding Rate Time Series (Sample)')
        axes[1, 0].set_ylabel('Funding Rate')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Majors vs Non-majors comparison
        majors = ['BTCUSDT', 'ETHUSDT']
        non_majors = [s for s in symbols if s not in majors]
        
        majors_coverage = [coverage_data[symbols.index(s)] for s in majors if s in symbols]
        non_majors_coverage = [coverage_data[symbols.index(s)] for s in non_majors if s in symbols]
        
        categories = ['Majors', 'Non-Majors']
        avg_coverage = [np.mean(majors_coverage) if majors_coverage else 0, 
                       np.mean(non_majors_coverage) if non_majors_coverage else 0]
        
        axes[1, 1].bar(categories, avg_coverage, color=['gold', 'lightgreen'], alpha=0.7)
        axes[1, 1].set_title('Average Coverage: Majors vs Non-Majors')
        axes[1, 1].set_ylabel('Coverage %')
        
        plt.tight_layout()
        plt.savefig('results/funding_analysis.png', dpi=300, bbox_inches='tight')
        print("Funding analysis visualization saved to: results/funding_analysis.png")
        
    def run_comprehensive_analysis(self):
        """Run all analysis components."""
        print("=" * 80)
        print("COMPREHENSIVE FUNDING RATE ANALYSIS")
        print("=" * 80)
        
        # Load data
        self.load_and_prepare_data()
        
        # Run all analyses
        signal_coverage = self.calculate_signal_coverage()
        funding_magnitude = self.analyze_funding_magnitude()
        strategy_performance = self.analyze_strategy_performance()
        majors_vs_non_majors = self.analyze_majors_vs_non_majors()
        
        # Create visualizations
        self.create_funding_visualizations()
        
        # Generate summary report
        self.generate_summary_report(signal_coverage, funding_magnitude, 
                                   strategy_performance, majors_vs_non_majors)
        
        return {
            'signal_coverage': signal_coverage,
            'funding_magnitude': funding_magnitude,
            'strategy_performance': strategy_performance,
            'majors_vs_non_majors': majors_vs_non_majors
        }
    
    def generate_summary_report(self, signal_coverage, funding_magnitude, 
                              strategy_performance, majors_vs_non_majors):
        """Generate comprehensive summary report."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        # Signal coverage summary
        avg_coverage = np.mean([data['coverage_pct'] for data in signal_coverage.values()])
        print(f"\n1. SIGNAL COVERAGE (16.5 bps threshold):")
        print(f"   Average coverage across all symbols: {avg_coverage:.2f}%")
        print(f"   Range: {min([data['coverage_pct'] for data in signal_coverage.values()]):.2f}% - "
              f"{max([data['coverage_pct'] for data in signal_coverage.values()]):.2f}%")
        
        # Funding magnitude summary
        avg_magnitude = np.mean([data['mean_abs_funding'] for data in funding_magnitude.values()])
        print(f"\n2. FUNDING MAGNITUDE:")
        print(f"   Average magnitude across all symbols: {avg_magnitude:.6f}")
        print(f"   Range: {min([data['mean_abs_funding'] for data in funding_magnitude.values()]):.6f} - "
              f"{max([data['mean_abs_funding'] for data in funding_magnitude.values()]):.6f}")
        
        # Strategy performance summary
        active_strategies = [data for data in strategy_performance.values() if data is not None]
        if active_strategies:
            avg_trade_count = np.mean([data['trade_count'] for data in active_strategies])
            avg_holding_period = np.mean([data['avg_holding_periods'] for data in active_strategies])
            print(f"\n3. STRATEGY PERFORMANCE:")
            print(f"   Average trade count per symbol: {avg_trade_count:.1f}")
            print(f"   Average holding periods: {avg_holding_period:.1f}")
        
        # Majors vs non-majors summary
        if 'majors' in majors_vs_non_majors and 'avg_coverage' in majors_vs_non_majors['majors']:
            print(f"\n4. MAJORS VS NON-MAJORS:")
            print(f"   Majors average coverage: {majors_vs_non_majors['majors']['avg_coverage']:.2f}%")
            print(f"   Non-majors average coverage: {majors_vs_non_majors['non_majors']['avg_coverage']:.2f}%")
        
        # Save detailed results
        self.save_detailed_results(signal_coverage, funding_magnitude, 
                                 strategy_performance, majors_vs_non_majors)
    
    def save_detailed_results(self, signal_coverage, funding_magnitude, 
                            strategy_performance, majors_vs_non_majors):
        """Save detailed results to CSV files."""
        
        # Signal coverage results
        coverage_df = pd.DataFrame.from_dict(signal_coverage, orient='index')
        coverage_df.to_csv('results/signal_coverage_analysis.csv')
        
        # Funding magnitude results
        magnitude_df = pd.DataFrame.from_dict(funding_magnitude, orient='index')
        magnitude_df.to_csv('results/funding_magnitude_analysis.csv')
        
        # Strategy performance results
        strategy_df = pd.DataFrame.from_dict(strategy_performance, orient='index')
        strategy_df.to_csv('results/strategy_performance_analysis.csv')
        
        print(f"\nDetailed results saved to:")
        print(f"  - results/signal_coverage_analysis.csv")
        print(f"  - results/funding_magnitude_analysis.csv")
        print(f"  - results/strategy_performance_analysis.csv")

if __name__ == "__main__":
    analyzer = FundingAnalyzer()
    results = analyzer.run_comprehensive_analysis()
