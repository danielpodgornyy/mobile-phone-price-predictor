import pandas as pd
from data_analyzer import DataAnalyzer
from model_grader import ModelGrader

def predict_mobile_phone_prices():
    data = pd.read_csv('./data/train.csv')

    da = DataAnalyzer(data)
    dp = ModelGrader(data)

    while True:
        print('')
        print('1. Descriptive statistics of mobile phone sample')
        print('2. Histogram of data distribution')
        print('3. Correlation heatmap')
        print('4. Confusion matrices (without feature selection)')
        print('5. Confusion matrices (with feature selection)')
        print('6. Exit')
        print('')

        val = input("Enter a digit:")

        match val:
            case '1':
                da.build_descriptive_data_table()
            case '2':
                da.build_histogram_of_data_distribution()
            case '3':
                da.build_correlation_heatmap()
            case '4':
                dp.predict_prices()
            case '5':
                dp.predict_prices(True)
            case '6':
                return

if __name__ == '__main__':
    predict_mobile_phone_prices()
