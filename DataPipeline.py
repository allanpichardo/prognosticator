import tensorflow as tf
import pandas as pd
import math
from yahooconverter import percent_b_indicator, pct_change, band_width_indicator

class DataPipeline:
    
    COLUMNS = [
        'adj_close',
        'percent_b',
        'band_width',
        'p_b_ratio',
        'p_e_ratio',
        'eps_diluted',
        'next_price'
    ]

    FIELD_DEFAULTS = [
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0],
    ]

    Y_COLUMN_NAME = "next_price"

    def __init__(self, training_csv_filepath, testing_csv_filepath):
        self.training_csv_filepath = training_csv_filepath
        self.testing_csv_filepath = testing_csv_filepath
    

    def _dataset_from_csv(self, csv_path):
        ds = tf.data.TextLineDataset(csv_path).skip(1)
        ds = ds.map(self._parse_line)
        return ds

    
    def get_class_as_string(self, value):
        if value == 2:
            return 'Buy'
        if value == 0:
            return 'Sell'
        else:
            return 'Hold'


    def training_input_fn(self, batch_size):
        ds = self._dataset_from_csv(self.training_csv_filepath)
        shuffle_amount = batch_size * 150
        ds = ds.shuffle(shuffle_amount).repeat().batch(batch_size)
        return ds.make_one_shot_iterator().get_next()

    
    def evaluate_input_fn(self, batch_size=None):
        dataset = self._dataset_from_csv(self.testing_csv_filepath)

        assert batch_size is not None, "batch_size must not be None"
        shuffle_amount = batch_size * 150
        dataset = dataset.shuffle(shuffle_amount).batch(batch_size)

        return dataset.make_one_shot_iterator().get_next()


    def _parse_line(self, line):
        fields = tf.decode_csv(line, self.FIELD_DEFAULTS)
        features = dict(zip(self.COLUMNS,fields))
        label = features.pop(self.Y_COLUMN_NAME)

        return features, label


    def _to_unix_timestamp(self, datetime):
        return datetime.value // 10 ** 9


    def _normalize(self, value, mean, std):
        return (value - mean) / std


    def _denormalize(self, value, mean, std):
        return std * value + mean

    def predict_input_fn(self, date, close_price, volume, sma_20, upper_bband, lower_bband, previous_price):
        date = pd.to_datetime(date)
        day_of_week = date.dayofweek
        day_of_year = date.dayofyear

        # inputs = {
        #     'adj_close' : [close_price],
        #     'volume' : [volume / 10 ** 6],
        #     'sma_20' : [self._normalize(sma_20, close_price, close_price)],
        #     'upper_band' : [self._normalize(upper_bband, close_price, close_price)],
        #     'lower_band' : [self._normalize(lower_bband, close_price, close_price)]
        # }
        inputs = {
            'adj_close' : [close_price],
            'percent_b' : [percent_b_indicator(close_price, upper_bband, lower_bband)],
            'band_width' : [band_width_indicator(upper_bband, lower_bband, sma_20)],
            'price_5_days_ago' : [previous_price]
        }

        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.batch(1)
        return dataset.make_one_shot_iterator().get_next()


    def get_feature_columns(self):
        feature_columns = []
        for name in self.COLUMNS[:-1]:
            feature_columns.append(tf.feature_column.numeric_column(name))
        
        return feature_columns
    
    def get_number_of_hidden_neurons(self):
        return int(math.ceil(len(self.COLUMNS) / 2))