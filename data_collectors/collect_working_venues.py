"""
Working multi-venue data collector.
Starts with Binance (which we know works) and can be extended with other venues.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from .load_binance_extended import BinanceExtendedCollector

logger = logging.getLogger(__name__)

class WorkingMultiVenueCollector:
    """Working collector for venues that are currently functional."""
    
    def __init__(self, data_dir: str = "raw_data"):
        self.data_dir = data_dir
        self.collectors = {
            'binance': BinanceExtendedCollector(data_dir)
        }
        self.unified_data = {}
        
    def test_all_connections(self) -> Dict[str, bool]:
        """Test connections to all working venues."""
        results = {}
        for venue, collector in self.collectors.items():
            logger.info(f"Testing connection to {venue}...")
            results[venue] = collector.test_connection()
        return results
    
    def collect_venue_data(self, venue: str, start_time: str, end_time: str, 
                          symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Collect data from a specific venue."""
        if venue not in self.collectors:
            logger.error(f"Unknown venue: {venue}")
            return {}
        
        collector = self.collectors[venue]
        
        if symbols:
            # Collect specific symbols
            results = {}
            for symbol in symbols:
                df = collector.collect_symbol_data(symbol, start_time, end_time)
                if df is not None:
                    results[symbol] = df
            return results
        else:
            # Collect all available symbols
            return collector.collect_all_symbols(start_time, end_time)
    
    def collect_major_symbols(self, start_time: str, end_time: str) -> Dict[str, pd.DataFrame]:
        """Collect data for major symbols across all working venues."""
        major_symbols = ['BTCUSDT', 'ETHUSDT', 'AAVEUSDT', 'UNIUSDT', 'COMPUSDT', 
                        'CRVUSDT', 'OPUSDT', 'ARBUSDT', 'DOGEUSDT', 'GALAUSDT', 'IMXUSDT']
        
        all_data = {}
        
        for venue, collector in self.collectors.items():
            logger.info(f"Collecting major symbols from {venue}...")
            venue_data = {}
            
            for symbol in major_symbols:
                df = collector.collect_symbol_data(symbol, start_time, end_time)
                if df is not None and not df.empty:
                    venue_data[symbol] = df
                    logger.info(f"  {symbol}: {len(df)} records")
                else:
                    logger.warning(f"  {symbol}: No data available")
            
            all_data[venue] = venue_data
        
        return all_data
    
    def create_unified_dataset(self, venues: List[str] = None) -> pd.DataFrame:
        """Create unified dataset from all working venues."""
        if venues is None:
            venues = list(self.collectors.keys())
        
        all_data = []
        
        for venue in venues:
            venue_dir = os.path.join(self.data_dir, venue)
            if not os.path.exists(venue_dir):
                logger.warning(f"Venue directory {venue_dir} not found")
                continue
            
            # Load all parquet files from venue directory
            for filename in os.listdir(venue_dir):
                if filename.endswith('.parquet'):
                    filepath = os.path.join(venue_dir, filename)
                    try:
                        df = pd.read_parquet(filepath)
                        all_data.append(df)
                        logger.info(f"Loaded {len(df)} records from {venue}/{filename}")
                    except Exception as e:
                        logger.error(f"Failed to load {filepath}: {str(e)}")
        
        if not all_data:
            logger.error("No data found to unify")
            return pd.DataFrame()
        
        # Combine all data
        unified_df = pd.concat(all_data, ignore_index=False)
        unified_df = unified_df.sort_index()
        
        # Save unified dataset
        unified_path = os.path.join(self.data_dir, "unified_data.parquet")
        unified_df.to_parquet(unified_path)
        
        logger.info(f"Created unified dataset with {len(unified_df)} records")
        logger.info(f"Venues: {unified_df['venue'].unique()}")
        logger.info(f"Assets: {unified_df['asset'].unique()}")
        logger.info(f"Sectors: {unified_df['sector'].unique()}")
        
        return unified_df
    
    def run_quality_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run quality checks on the unified dataset."""
        checks = {}
        
        # 1. Coverage check
        coverage_by_venue = df.groupby('venue').size()
        coverage_by_asset = df.groupby('asset').size()
        checks['coverage_by_venue'] = coverage_by_venue.to_dict()
        checks['coverage_by_asset'] = coverage_by_asset.to_dict()
        
        # 2. Funding rate sanity check
        funding_rates = df['funding_rate'].dropna()
        if not funding_rates.empty:
            checks['funding_stats'] = {
                'median': float(funding_rates.median()),
                'mean': float(funding_rates.mean()),
                'std': float(funding_rates.std()),
                'min': float(funding_rates.min()),
                'max': float(funding_rates.max()),
                'count': len(funding_rates)
            }
        
        # 3. Price correlation check (index vs mark)
        price_corr = df[['mark_c', 'index_c']].corr().iloc[0, 1]
        checks['price_correlation'] = float(price_corr) if not pd.isna(price_corr) else None
        
        # 4. Data completeness
        checks['completeness'] = {
            'total_records': len(df),
            'missing_funding': df['funding_rate'].isna().sum(),
            'missing_prices': df['mark_c'].isna().sum(),
            'missing_index': df['index_c'].isna().sum()
        }
        
        return checks
    
    def generate_summary_report(self, df: pd.DataFrame, checks: Dict[str, Any]) -> str:
        """Generate a summary report of the data collection."""
        report = []
        report.append("=" * 80)
        report.append("MULTI-VENUE DATA COLLECTION SUMMARY")
        report.append("=" * 80)
        report.append(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Records: {len(df):,}")
        report.append("")
        
        # Venue breakdown
        report.append("VENUE BREAKDOWN:")
        for venue, count in checks['coverage_by_venue'].items():
            report.append(f"  {venue.upper()}: {count:,} records")
        report.append("")
        
        # Asset breakdown
        report.append("ASSET BREAKDOWN:")
        for asset, count in checks['coverage_by_asset'].items():
            report.append(f"  {asset}: {count:,} records")
        report.append("")
        
        # Funding rate analysis
        if 'funding_stats' in checks:
            stats = checks['funding_stats']
            report.append("FUNDING RATE ANALYSIS:")
            report.append(f"  Median: {stats['median']:.6f} ({stats['median']*10000:.2f} bps)")
            report.append(f"  Mean: {stats['mean']:.6f} ({stats['mean']*10000:.2f} bps)")
            report.append(f"  Std Dev: {stats['std']:.6f} ({stats['std']*10000:.2f} bps)")
            report.append(f"  Range: {stats['min']:.6f} to {stats['max']:.6f}")
            report.append(f"  Records: {stats['count']:,}")
            report.append("")
        
        # Data quality
        report.append("DATA QUALITY:")
        completeness = checks['completeness']
        report.append(f"  Total Records: {completeness['total_records']:,}")
        report.append(f"  Missing Funding Rates: {completeness['missing_funding']:,}")
        report.append(f"  Missing Mark Prices: {completeness['missing_prices']:,}")
        report.append(f"  Missing Index Prices: {completeness['missing_index']:,}")
        
        if checks['price_correlation'] is not None:
            report.append(f"  Mark-Index Correlation: {checks['price_correlation']:.4f}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if checks['price_correlation'] and checks['price_correlation'] < 0.99:
            report.append("  ⚠️  Low mark-index correlation - check price mapping")
        if completeness['missing_funding'] > completeness['total_records'] * 0.1:
            report.append("  ⚠️  High missing funding rate data - check API endpoints")
        if completeness['missing_prices'] > completeness['total_records'] * 0.05:
            report.append("  ⚠️  High missing price data - check price endpoints")
        
        if all([
            checks['price_correlation'] and checks['price_correlation'] > 0.99,
            completeness['missing_funding'] < completeness['total_records'] * 0.05,
            completeness['missing_prices'] < completeness['total_records'] * 0.02
        ]):
            report.append("  ✅ Data quality looks good for backtesting")
        
        return "\n".join(report)

def test_working_collector():
    """Test the working multi-venue collector."""
    collector = WorkingMultiVenueCollector()
    
    # Test connection
    logger.info("Testing connections...")
    connections = collector.test_all_connections()
    for venue, success in connections.items():
        logger.info(f"  {venue}: {'✅' if success else '❌'}")
    
    if not any(connections.values()):
        logger.error("No venues connected successfully")
        return False
    
    # Test data collection
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    logger.info(f"Collecting data from {start_time.date()} to {end_time.date()}")
    data = collector.collect_major_symbols(start_time.isoformat(), end_time.isoformat())
    
    if not data:
        logger.error("No data collected")
        return False
    
    # Create unified dataset
    logger.info("Creating unified dataset...")
    unified_df = collector.create_unified_dataset()
    
    if unified_df.empty:
        logger.error("Failed to create unified dataset")
        return False
    
    # Run quality checks
    logger.info("Running quality checks...")
    checks = collector.run_quality_checks(unified_df)
    
    # Generate report
    report = collector.generate_summary_report(unified_df, checks)
    logger.info("\n" + report)
    
    # Save report
    report_path = os.path.join(collector.data_dir, "data_collection_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to: {report_path}")
    return True

if __name__ == "__main__":
    test_working_collector()
