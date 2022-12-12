# Financial-Clustering of SP500 Companies

Experimenting with different clustering algorithms to build portfolios based on SP500 index companies.

## IPYNB Files Info (run in this order)
1. "Yfinance Rets.ipynb" - Downloads/parses daily data of SP500 companies. Calculates/exports Sharpe ratios.
2. "FMC Fundamentals.ipynb" - Downloads/parses quarterly filings of SP500 companies
3. "Detrending_OCO.ipynb" - Detrends $\frac{(Open - Close)}{Open}$ version of Daily Data. Exports numerous versions of detrended data.
4. "Detrending_Pct_Change.ipynb" - Detrends $%change$ version of Daily Data. Exports numerous versions of detrended data.
5. "clustering_algos.ipynb" - Uses exports from preceeding 4 files to tune hyperparameters and determine optimal portfolio of SP500 companies. Clustering algorithms used are K-means, OptiScan, DBScan

## Other Files Info
1. "DTB3.csv" - Three Month Treasury Bill Yields. 
2. "S&P 500 Historical Components & Changes(10-14-2022).csv" - List of all SP500 constituents for each day there was a change to the list of SP500 companies(company could be added or dropped).
3. "quarterly_balance" - Pickle dump of quarterly balance sheet data downloaded using the Financial Modelling Prep company endpoint.
4. "quarterly_income" - Pickle dump of quarterly income statement data downloaded using the Financial Modelling Prep company endpoint.
5. "sp500_historical.csv" - Historical Daily data for SP500 ticker.
6. "Exports" Folder - Exports from all IPYNB files.
7. "sp500change.xlsx" - Log of changes to SP500 constituent companies(deprecated)
8. "yakPak.py" - Package for common functions (barely used, needs updating)

## Credits
* Loosely based on the following paper: [K-Means Stock Clustering Analysis Based on Historical Price
Movements and Financial Ratios](https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=3517&context=cmc_theses)
* Financial Modelling Prep Company
