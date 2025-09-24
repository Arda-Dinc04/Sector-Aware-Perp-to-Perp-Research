#!/usr/bin/env python3
"""
Simple API connection test for all venues.
Tests basic connectivity without attempting data collection.
"""

import sys
import os
import logging
import requests
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dydx_connection():
    """Test dYdX API connection."""
    try:
        url = "https://api.dydx.exchange/v3/markets"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        markets_count = len(data.get('markets', {}))
        logger.info(f"✅ dYdX: Connected successfully, found {markets_count} markets")
        return True
    except Exception as e:
        logger.error(f"❌ dYdX: Connection failed - {str(e)}")
        return False

def test_gmx_connection():
    """Test GMX subgraph connection."""
    try:
        url = "https://api.thegraph.com/subgraphs/name/gmx-io/gmx-arbitrum-stats"
        query = "{ _meta { block { number } } }"
        response = requests.post(url, json={'query': query}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'data' in data and '_meta' in data['data']:
            block_number = data['data']['_meta']['block']['number']
            logger.info(f"✅ GMX: Connected successfully, block {block_number}")
            return True
        else:
            logger.error(f"❌ GMX: Unexpected response format")
            return False
    except Exception as e:
        logger.error(f"❌ GMX: Connection failed - {str(e)}")
        return False

def test_drift_connection():
    """Test Drift API connection."""
    try:
        # Try multiple possible endpoints
        endpoints = [
            "https://api.drift.trade/api/markets",
            "https://api.drift.trade/markets", 
            "https://drift-api.vercel.app/api/markets"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ Drift: Connected successfully to {endpoint}")
                    return True
            except:
                continue
        
        logger.error("❌ Drift: All endpoints failed")
        return False
    except Exception as e:
        logger.error(f"❌ Drift: Connection failed - {str(e)}")
        return False

def test_binance_connection():
    """Test Binance API connection (our working baseline)."""
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        symbols_count = len(data.get('symbols', []))
        logger.info(f"✅ Binance: Connected successfully, found {symbols_count} symbols")
        return True
    except Exception as e:
        logger.error(f"❌ Binance: Connection failed - {str(e)}")
        return False

def main():
    """Main test function."""
    logger.info("=" * 80)
    logger.info("API CONNECTION TEST")
    logger.info("=" * 80)
    logger.info(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Test all APIs
    results = {}
    results['dydx'] = test_dydx_connection()
    results['gmx'] = test_gmx_connection()
    results['drift'] = test_drift_connection()
    results['binance'] = test_binance_connection()
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("CONNECTION TEST SUMMARY")
    logger.info("=" * 80)
    
    working_apis = []
    for api, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        logger.info(f"{api.upper():<10}: {status}")
        if success:
            working_apis.append(api)
    
    logger.info("")
    logger.info(f"Working APIs: {len(working_apis)}/4")
    
    if working_apis:
        logger.info(f"Available for data collection: {', '.join(working_apis)}")
    else:
        logger.warning("No APIs are working - check network connectivity")
    
    return len(working_apis) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
