# Frozen macro panels -- Table 4 run (LOCKED)
Exact copies of latent_state_estimation/*.csv from git commit a6c4f3f (2026-08-23), the data behind Table 4.
These files are the only macro source: downloading is disabled (macro_importer.py and import_data.py stop at once) and
explore/build_conditioning_sets.py builds each conditioning set once from them (daily panels cut to the config window).
Monthly: through 2026-07-01 (pi_transfer, personal_consump through 2026-06; real_manf_trade through 2026-05).
Daily growth: 2000-09-25 .. 2026-08-20. Daily inflation: 2000-08-31 .. 2026-08-21.
Resulting state (git a6c4f3f explore/macro_data_new.csv): m_t 2000-09-25 .. 2026-08-20, 6,350 days.

## stock_log_returns.csv (LOCKED 2026-11-10)
Daily log returns read by explore/build_conditioning_sets.py; yfinance is never called.
2000-09-25 .. 2026-08-20, 6,514 days, 15 columns.
- IBM CSCO AAPL MSFT ORCL INTC TXN QCOM AMAT ADBE: bit-identical to git a6c4f3f explore/macro_data_new.csv (the Table 4 returns).
- KO PG CL GIS KMB: not part of the Table 4 run (no August data exists); frozen from the 2026-11-07 yfinance download.
