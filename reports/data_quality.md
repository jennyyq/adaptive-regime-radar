# Data quality report

Generated from raw files last updated 2026-10-20T14:59:38+00:00 UTC.

## Investment panel (9 ETFs)

- Common sample: **2007-05-30 → 2026-10-19** (4879 price rows, 4878 return rows)
- Binding start ETF: BIL (first trade 2007-05-30)
- Missing daily returns inside the common sample (left as NaN): 0

## Flags for review

- EEM: 1 |log return| > 20% (2008-10-13) — check against a second source
- BIL: adjusted close unchanged for 19 days
- XLE: 1 |log return| > 20% (2020-03-09) — check against a second source
- ^VIX: 122 |log return| > 20% (2006-05-30;2006-06-15;2006-07-13;2007-02-27;2007-03-06) — check against a second source
- ^VIX3M: 27 |log return| > 20% (2007-02-27;2010-05-07;2010-05-10;2011-08-04;2011-08-08) — check against a second source
- ^MOVE: 94 missing trading days after first listing
- ^MOVE: 17 |log return| > 20% (2004-12-08;2007-06-12;2007-08-09;2008-09-15;2013-08-02) — check against a second source
- FRED NFCI: some release dates are rule-based approximations (vintage:first_release=802; fallback:pre_vintage_history=1)

## Yahoo coverage

| ticker | required | first_date | last_date | observations | missing_in_live_range | coverage_in_live_range_pct | off_calendar_rows | non_positive_prices | extreme_returns | longest_unchanged_run | zero_volume_days | inconsistent_ohlc_bars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPY | True | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 2 | 0 | 0 |
| EFA | True | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 3 | 0 | 0 |
| EEM | True | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 1 | 2 | 0 | 0 |
| IEF | True | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 3 | 0 | 0 |
| TIP | True | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 3 | 0 | 0 |
| HYG | True | 2007-04-11 | 2026-10-19 | 4913 | 0 | 100.0 | 0 | 0 | 0 | 4 | 0 | 0 |
| GLD | True | 2004-11-18 | 2026-10-19 | 5513 | 0 | 100.0 | 0 | 0 | 0 | 2 | 0 | 0 |
| DBC | True | 2006-02-06 | 2026-10-19 | 5208 | 0 | 100.0 | 0 | 0 | 0 | 3 | 0 | 0 |
| BIL | True | 2007-05-30 | 2026-10-19 | 4879 | 0 | 100.0 | 0 | 0 | 0 | 19 | 0 | 0 |
| QQQ | True | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 3 | 0 | 0 |
| IWM | True | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 2 | 0 | 0 |
| RSP | False | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 2 | 0 | 0 |
| LQD | True | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 2 | 0 | 0 |
| XLB | False | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 3 | 0 | 0 |
| XLE | False | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 1 | 3 | 0 | 0 |
| XLF | False | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 3 | 0 | 0 |
| XLI | False | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 3 | 0 | 0 |
| XLK | False | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 2 | 0 | 0 |
| XLP | False | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 3 | 0 | 0 |
| XLU | False | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 2 | 0 | 0 |
| XLV | False | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 3 | 0 | 0 |
| XLY | False | 2004-01-02 | 2026-10-19 | 5735 | 0 | 100.0 | 0 | 0 | 0 | 2 | 0 | 0 |
| ^VIX | True | 2004-01-02 | 2026-10-19 | 5737 | 0 | 100.0 | 2 | 0 | 122 | 3 | 0 | 0 |
| ^VIX3M | False | 2006-07-17 | 2026-10-19 | 5097 | 0 | 100.0 | 0 | 0 | 27 | 3 | 0 | 0 |
| ^MOVE | False | 2004-01-02 | 2026-10-19 | 5641 | 94 | 98.36 | 0 | 0 | 17 | 3 | 0 | 0 |

## FRED coverage (point-in-time)

| series_id | status | frequency | first_obs | last_obs | observations | median_release_lag_days | first_usable_trade_date | max_staleness_days |
|---|---|---|---|---|---|---|---|---|
| DGS3MO | ok | daily | 2004-01-02 | 2026-10-16 | 5702 | 1.0 | 2004-01-05 | 4 |
| DGS2 | ok | daily | 2004-01-02 | 2026-10-16 | 5702 | 1.0 | 2004-01-05 | 4 |
| DGS10 | ok | daily | 2004-01-02 | 2026-10-16 | 5702 | 1.0 | 2004-01-05 | 4 |
| T10Y2Y | ok | daily | 2004-01-02 | 2026-10-19 | 5703 | 1.0 | 2004-01-05 | 4 |
| T10Y3M | ok | daily | 2004-01-02 | 2026-10-19 | 5703 | 1.0 | 2004-01-05 | 4 |
| DFII10 | ok | daily | 2004-01-02 | 2026-10-16 | 5702 | 1.0 | 2004-01-05 | 4 |
| T10YIE | ok | daily | 2004-01-02 | 2026-10-19 | 5703 | 1.0 | 2004-01-05 | 4 |
| T5YIE | ok | daily | 2004-01-02 | 2026-10-19 | 5703 | 1.0 | 2004-01-05 | 4 |
| BAA10Y | ok | daily | 2004-01-02 | 2026-10-16 | 5697 | 1.0 | 2004-01-05 | 4 |
| BAMLH0A0HYM2 | ok | daily | 2023-10-23 | 2026-10-19 | 784 | 1.0 | 2023-10-24 | 4 |
| NFCI | ok | weekly | 2011-05-27 | 2026-10-09 | 803 | 5.0 | 2011-06-01 | 26 |
| CPIAUCSL | ok | monthly | 2004-01-01 | 2026-09-01 | 272 | 44.0 | 2004-02-20 | 107 |
| CPILFESL | ok | monthly | 2004-01-01 | 2026-09-01 | 272 | 44.0 | 2004-02-20 | 107 |
