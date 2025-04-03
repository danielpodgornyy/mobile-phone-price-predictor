import pandas as pd
from data_analyzer import DataAnalyzer
from model_grader import ModelGrader

def predict_mobile_phone_prices():
    data = pd.read_csv('./data/train.csv')

    da = DataAnalyzer(data)
    da.build_descriptive_data_table()
    #da.build_histogram_of_data_distribution()
    #da.build_correlation_heatmap()

    dp = ModelGrader(data)
    #dp.predict_prices()


if __name__ == '__main__':
    predict_mobile_phone_prices()
