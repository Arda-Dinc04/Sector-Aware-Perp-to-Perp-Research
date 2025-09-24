#!/usr/bin/env python3
"""
Test script for multi-venue data collection.
Tests each venue individually to verify data collection works correctly.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collectors.collect_all_venues import test_single_venue, test_all_venues

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main test function."""
    logger.info("=" * 80)
    logger.info("MULTI-VENUE DATA COLLECTION TEST")
    logger.info("=" * 80)
    
    # Test dYdX first (most reliable)
    logger.info("\n1. Testing dYdX data collection...")
    dydx_success = test_single_venue("dydx")
    
    if dydx_success:
        logger.info("✅ dYdX test passed")
    else:
        logger.error("❌ dYdX test failed")
    
    # Test GMX
    logger.info("\n2. Testing GMX data collection...")
    gmx_success = test_single_venue("gmx")
    
    if gmx_success:
        logger.info("✅ GMX test passed")
    else:
        logger.error("❌ GMX test failed")
    
    # Test Drift
    logger.info("\n3. Testing Drift data collection...")
    drift_success = test_single_venue("drift")
    
    if drift_success:
        logger.info("✅ Drift test passed")
    else:
        logger.error("❌ Drift test failed")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"dYdX: {'✅ PASS' if dydx_success else '❌ FAIL'}")
    logger.info(f"GMX:  {'✅ PASS' if gmx_success else '❌ FAIL'}")
    logger.info(f"Drift: {'✅ PASS' if drift_success else '❌ FAIL'}")
    
    total_success = sum([dydx_success, gmx_success, drift_success])
    logger.info(f"\nTotal: {total_success}/3 venues working")
    
    if total_success > 0:
        logger.info("✅ At least one venue is working - data collection setup is functional")
    else:
        logger.error("❌ No venues are working - check API endpoints and network connectivity")
    
    return total_success > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
