#!/usr/bin/env python3
"""
Main execution script for baseline backtesting.

Runs both baseline strategies and generates comprehensive visualizations.
"""

import os
import sys
from datetime import datetime
from backtester import Backtester, BacktesterConfig
from visualizer import BaselineVisualizer

def main():
    """Main execution function."""
    print("=" * 80)
    print("SECTOR-AWARE PERP-TO-PERP RESEARCH")
    print("BASELINE STRATEGIES BACKTEST")
    print("=" * 80)
    print(f"Execution started at: {datetime.now()}")
    print()
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize configuration
    config = BacktesterConfig()
    
    # Print configuration
    print("CONFIGURATION:")
    print(f"  Data Directory: {config.data_dir}")
    print(f"  Funding Window: {config.funding_window}")
    print(f"  Notional per Symbol: ${config.notional_usd:,}")
    print(f"  Leverage: {config.leverage}x")
    print(f"  Spot Fees: {config.fees['spot_taker_bps']} bps")
    print(f"  Perp Fees: {config.fees['perp_taker_bps']} bps")
    print(f"  Epsilon Buffer: {config.epsilon_bps} bps")
    print(f"  Basis Stop: {config.basis_stop_bps} bps")
    print()
    
    try:
        # Initialize and run backtester
        backtester = Backtester(config)
        results = backtester.run_baselines()
        
        # Create visualizations
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        visualizer = BaselineVisualizer(results, config)
        visualizer.create_all_plots(save_dir=output_dir)
        
        # Save results summary
        summary_file = os.path.join(output_dir, "backtest_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("SECTOR-AWARE PERP-TO-PERP RESEARCH - BASELINE BACKTEST\n")
            f.write("=" * 60 + "\n")
            f.write(f"Execution completed at: {datetime.now()}\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"  Data Directory: {config.data_dir}\n")
            f.write(f"  Funding Window: {config.funding_window}\n")
            f.write(f"  Notional per Symbol: ${config.notional_usd:,}\n")
            f.write(f"  Leverage: {config.leverage}x\n")
            f.write(f"  Spot Fees: {config.fees['spot_taker_bps']} bps\n")
            f.write(f"  Perp Fees: {config.fees['perp_taker_bps']} bps\n")
            f.write(f"  Epsilon Buffer: {config.epsilon_bps} bps\n")
            f.write(f"  Basis Stop: {config.basis_stop_bps} bps\n\n")
            
            f.write("RESULTS SUMMARY:\n")
            for key, result in results.items():
                if isinstance(result, dict) and 'strategy' in result:
                    f.write(f"\n{result['strategy']} ({key}):\n")
                    f.write(f"  Total Return: {result['total_return']:.2%}\n")
                    f.write(f"  Annualized Return: {result['annualized_return']:.2%}\n")
                    f.write(f"  Volatility: {result['volatility']:.2%}\n")
                    f.write(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}\n")
                    f.write(f"  Max Drawdown: {result['max_drawdown']:.2%}\n")
                    f.write(f"  Fees Paid: ${result['fees_paid']:.2f}\n")
                    if 'funding_pnl' in result:
                        f.write(f"  Funding PnL: ${result['funding_pnl']:.2f}\n")
                        if 'basis_pnl' in result:
                            f.write(f"  Basis PnL: ${result['basis_pnl']:.2f}\n")
        
        print(f"\nResults saved to: {output_dir}/")
        print(f"Summary saved to: {summary_file}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("BACKTEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Quick performance comparison
        if 'buy_hold' in results and 'hedged_carry' in results:
            bh_return = results['buy_hold']['total_return']
            carry_return = results['hedged_carry']['total_return']
            bh_sharpe = results['buy_hold']['sharpe_ratio']
            carry_sharpe = results['hedged_carry']['sharpe_ratio']
            
            print(f"\nQUICK COMPARISON (All Symbols):")
            print(f"  Buy & Hold:     {bh_return:.2%} return, {bh_sharpe:.2f} Sharpe")
            print(f"  Hedged Carry:   {carry_return:.2%} return, {carry_sharpe:.2f} Sharpe")
            
            if carry_return > bh_return:
                print(f"  Hedged Carry outperformed by: {(carry_return - bh_return):.2%}")
            else:
                print(f"  Buy & Hold outperformed by: {(bh_return - carry_return):.2%}")
        
        return results
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Backtest failed. Please check the error and try again.")
        return None

if __name__ == "__main__":
    results = main()
