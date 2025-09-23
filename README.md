# Binance Futures Data Collector

A Python script to collect comprehensive cryptocurrency futures data from Binance USDⓈ-M Futures API at 30-minute resolution for approximately 3 years of history.

## Features

- **Multi-asset data collection** across crypto sectors (Majors, DeFi, Layer2, AI/Gaming, Memes)
- **30-minute resolution** for all data points
- **Comprehensive data sources**:
  - Funding rates (resampled from 8h to 30m with accrual calculation)
  - Mark price klines (OHLC)
  - Index price klines (OHLC)
  - Futures price klines (OHLC + volume)
  - Open interest history (~30 days due to API limitations)
- **Rate limiting protection** to stay within Binance API limits
- **Robust error handling** with retry logic
- **Parquet output** for efficient storage and analysis

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the data collector:
```bash
python pull_binance_30m.py
```

## Important: Regional Restrictions

⚠️ **Binance Futures API may be restricted in certain regions**. If you encounter HTTP 451 errors with the message "Service unavailable from a restricted location", you will need to:

- Run the script from a non-restricted region
- Use a VPN to access from an allowed location
- Consider alternative data sources for restricted regions

The script will work correctly once API access is available.

## Data Schema

The output Parquet files contain the following columns:

| Column | Description |
|--------|-------------|
| `timestamp` | UTC timestamp (30m intervals) |
| `exchange` | Always "Binance" |
| `symbol` | Trading pair (e.g., "BTCUSDT") |
| `sector` | Asset sector classification |
| `funding_rate_interval` | Native 8h funding rate |
| `funding_rate_30m` | Prorated 30m funding accrual |
| `mark_open/high/low/close` | Mark price OHLC |
| `index_open/high/low/close` | Index price OHLC |
| `price_open/high/low/close` | Futures price OHLC |
| `volume` | Trading volume (base units) |
| `quoteVolume` | Trading volume (USDT) |
| `open_interest` | Open interest in USD (last ~30 days only) |

## Sector Classification

The script collects data for the following crypto sectors:

- **Majors**: BTC, ETH
- **DeFi**: UNI, AAVE, CRV, COMP
- **Layer2_Infra**: ARB, OP, MATIC (if available)
- **AI_Gaming**: IMX, RNDR, AGIX, GALA (if available)
- **Memes**: DOGE, SHIB, PEPE

*Note: Some symbols (RNDRUSDT, MATICUSDT, AGIXUSDT) may not be available in all environments and will be automatically skipped.*

## Rate Limiting

The script implements conservative rate limiting:
- Target: ≤1,500 request weights/minute (well under Binance's 2,400 limit)
- Inter-request delays: 100ms minimum
- Automatic backoff on HTTP 429 responses
- Weight monitoring via API response headers

## Data Constraints

### Open Interest History
- **Limitation**: Binance only provides ~30 days of open interest history via public REST API
- **Impact**: Older periods will have `NaN` values for `open_interest`
- **Alternative**: For longer OI history, consider supplementing with third-party data providers

### Funding Rate Resampling
- **Native interval**: 8 hours
- **Resampling method**: Forward-fill with proportional accrual calculation
- **Formula**: `funding_rate_30m = funding_rate_8h × (30min / 480min)`

## Output Structure

Data is saved to `data/binance_30m/` with one Parquet file per symbol:
```
data/binance_30m/
├── BTCUSDT.parquet
├── ETHUSDT.parquet
├── UNIUSDT.parquet
└── ...
```

## Usage Examples

### Basic Collection
```bash
python pull_binance_30m.py
```

### Loading Data for Analysis
```python
import pandas as pd

# Load specific symbol
btc_data = pd.read_parquet('data/binance_30m/BTCUSDT.parquet')

# Load all data
import glob
all_files = glob.glob('data/binance_30m/*.parquet')
combined_data = pd.concat([pd.read_parquet(f) for f in all_files])
```

## API Endpoints Used

1. `/fapi/v1/exchangeInfo` - Symbol availability
2. `/fapi/v1/fundingRate` - Funding rate history
3. `/fapi/v1/markPriceKlines` - Mark price data
4. `/fapi/v1/indexPriceKlines` - Index price data
5. `/fapi/v1/klines` - Futures price and volume data
6. `/futures/data/openInterestHist` - Open interest data

## Logging

The script provides detailed logging for:
- Data collection progress
- Rate limiting activities
- API errors and retries
- Data quality checks
- File save confirmations

## Contributing

When modifying the script, ensure:
- Rate limiting logic remains conservative
- Error handling covers network failures
- Data schema consistency is maintained
- Parquet compression settings are preserved