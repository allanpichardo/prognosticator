from DataPipeline import DataPipeline

pipeline = DataPipeline('.csv/train.csv', '.csv/test.csv')

print(pipeline.training_input_fn(batch_size=20))

print(pipeline.evaluate_input_fn(batch_size=20))

print(pipeline.predict_input_fn('04-03-2018', 9.54, 150000, 9.4310, 0.4838670978905855, 0.01))