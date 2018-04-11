import sys
import os
import glob
import pandas as pd
import numpy as np


CLASS_THRESHOLD = 0.1


def normalize(series, mean, std):
    return (series - mean) / std


def pct_change(present_close, future_close):
    return (future_close - present_close) / present_close


def _map_pct_to_action_class(value, threshold=CLASS_THRESHOLD):
    if value >= threshold:
        return 2
    elif value <= (threshold * -1):
        return 0
    else:
        return 1


def calculate_action_class(closing_series, future_series):
    pct_series = pct_change(closing_series, future_series)
    return pct_series.map(_map_pct_to_action_class)


def percent_b_indicator(closing_series, upper_band_series, lower_band_series):
    # %B = (Price - Lower Band)/(Upper Band - Lower Band)
    return (closing_series - lower_band_series) / (upper_band_series - lower_band_series)


def band_width_indicator(upper_band, lower_band, moving_average):
    # ( (Upper Band - Lower Band) / Middle Band) * 100
    return ((upper_band - lower_band) / moving_average) * 100


def stockpup_csv_to_dataframe(file_path):
    print('Processing quarterly earnings data.')

    quarterly = pd.read_csv(file_path)
    quarterly = quarterly[['Quarter end', 'P/B ratio', 'P/E ratio', 'EPS diluted']]
    quarterly = quarterly.rename(columns={'Quarter end':'timestamp', 'P/B ratio': 'p_b_ratio', 'P/E ratio':'p_e_ratio', 'EPS diluted':'eps_diluted'})
    quarterly = quarterly.replace('None', 0)
    quarterly['timestamp'] = pd.to_datetime(quarterly['timestamp'])
    quarterly = quarterly.set_index('timestamp')

    return quarterly.iloc[::-1]


def combine_csvs_in_path(path):
    path = r'{}'.format(path)
    all_files = glob.glob(os.path.join(path, "*.csv")) 
    return pd.concat((pd.read_csv(f) for f in all_files))


def create_composite_dataframe(yahoo, stockpup):
    # combine the frames, filter out the nans and compute future price
    stockpup = stockpup.shift(1, freq=pd.offsets.BDay())
    combined = pd.concat([yahoo, stockpup], axis=1)
    combined = combined.dropna()
    
    combined['next_price'] = combined['adj_close'].shift(-1)
    combined['next_price'] = calculate_action_class(combined['adj_close'], combined['next_price'])
    combined = combined.iloc[:-1]

    return combined


def yahoo_csv_to_dataframe(file_path):
    print('Processing price data.')
    raw_amd = pd.read_csv(file_path)

    if len(raw_amd) <=25:
        print('Error: this data is not long enough. It needs to be more than 25 records.')
        sys.exit()

    raw_amd.rename(columns={'Adj Close': 'adj_close', 'Volume': 'volume', 'Date': 'timestamp'}, inplace=True)

    print('Computing indicators')
    raw_amd['timestamp'] = pd.to_datetime(raw_amd['timestamp'])
    # raw_amd['year'] = raw_amd['timestamp'].dt.year
    # raw_amd['day_of_week'] = raw_amd['timestamp'].dt.dayofweek
    # raw_amd['day_of_year'] = raw_amd['timestamp'].dt.dayofyear

    mean = raw_amd['adj_close'].mean()
    std = raw_amd['adj_close'].std()
    # vol_mean = raw_amd['volume'].mean()
    # vol_std = raw_amd['volume'].std()

    raw_amd['adj_close'] = raw_amd['adj_close']
    raw_amd['volume'] = raw_amd['volume'] / 10 ** 6
    raw_amd['sma_20'] = raw_amd['adj_close'].rolling(window=20).mean()
    rolling_std = raw_amd['adj_close'].rolling(window=20).std()
    raw_amd['upper_band'] = raw_amd['sma_20'] + (rolling_std * 2)
    raw_amd['lower_band'] = raw_amd['sma_20'] - (rolling_std * 2)

    # raw_amd['timestamp'] = raw_amd['timestamp'].values.astype(np.int64) // 10 ** 9

    corpus = raw_amd.drop(labels=['volume', 'Open', 'High', 'Low', 'Close'], axis=1)
    corpus = corpus.set_index('timestamp')
    corpus = corpus.iloc[19:]

    print('Normalizing...')
    # corpus['percent_b'] = percent_b_indicator(corpus['adj_close'], corpus['upper_band'], corpus['lower_band'])
    # corpus['band_width'] = band_width_indicator(corpus['upper_band'], corpus['lower_band'], corpus['sma_20'])
    # corpus['sma_20'] = pct_change(corpus['adj_close'], corpus['sma_20'])
    # corpus['upper_band'] = pct_change(corpus['adj_close'], corpus['upper_band'])
    # corpus['lower_band'] = pct_change(corpus['adj_close'], corpus['lower_band'])

    # corpus = corpus.drop(labels=['adj_close'], axis=1)

    print('Aligning with prediction')
    # corpus['price_5_days_ago'] = corpus['adj_close'].shift(5)
    # corpus = corpus.iloc[5:]
    
    # corpus['price_in_5_days'] = corpus['adj_close'].shift(-5)
    # corpus = corpus.iloc[:-5]

    # corpus['price_in_5_days'] = calculate_action_class(corpus['adj_close'], corpus['price_in_5_days'])

    corpus = corpus.drop(labels=['sma_20', 'upper_band', 'lower_band'], axis=1)

    return corpus


def dataframe_to_csv(dataframe, outfile):
    dataframe.to_csv(path_or_buf=outfile, index=False)


if len(sys.argv) == 4:
    yahoofile = sys.argv[1]
    spfile = sys.argv[2]
    outfile = sys.argv[3]

    print('Loading {}...'.format(yahoofile))
    yahoo = yahoo_csv_to_dataframe(yahoofile)
    stockpup = stockpup_csv_to_dataframe(spfile)
    df = create_composite_dataframe(yahoo, stockpup)
    print('Saving to {}'.format(outfile))
    dataframe_to_csv(dataframe=df, outfile=outfile)
    print('Done!')
elif len(sys.argv) == 3:
    if sys.argv[1] == 'combine':
        path = sys.argv[2]
        print('Combining files in path {}'.format(path))
        df = combine_csvs_in_path(path)
        outfile = '{}/combined.csv'.format(path)
        dataframe_to_csv(df, outfile)
else:
    print("Missing infile and outfile")