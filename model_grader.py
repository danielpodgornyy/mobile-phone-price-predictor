# Imports for running models
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as NB

# Imports for plotting
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class ModelGrader():
    def __init__(self, data):
        self.data = data

    def predict_prices(self, feature_selection=False):
        if (feature_selection):
            x = pd.concat([self.data['ram'], self.data['battery_power'], self.data['px_width'], self.data['px_height']], axis=1)
        else:
            x = self.data.drop(columns=['price_range'])

        y = self.data['price_range']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        fig, ax = plt.subplots(2, 2)

        models = [SVM(), DT(), KNN(), NB()]
        model_names = ['SVM', 'DT', 'KNN', 'NB']
        reports = []

        i, j = 0, 0
        for model, model_name in zip(models, model_names):
            y_pred = model.fit(x_train, y_train).predict(x_test)

            self.plot_model(model_name, i, j, y_test, y_pred, ax)

            report = classification_report(y_test, y_pred, output_dict=True, target_names=['0', '1', '2', '3'])
            reports.append(report)

            j += 1
            if j >= 2:
                i += 1
                j = 0

            if i >= 2:
                break

        self.add_table(model_names, reports)

        plt.show()

    def plot_model(self, model_name, row, column, y_test, y_pred, ax):
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2', '3'])
        disp.plot(cmap=plt.cm.Blues, ax=ax[row, column])

        ax[row, column].set_title(f'{model_name} Confusion Matrix', fontsize=8)
        ax[row, column].set_xlabel('Prediction', fontsize=8)
        ax[row, column].set_ylabel('Actual', fontsize=8)

        ax[row, column].xaxis.set_label_position('top')

        plt.subplots_adjust(bottom=0.2, right=0.8, top=0.8, hspace=0.4)  # Adjust subplots to fit title and labels

    def add_table(self, model_names, reports):
        values = [
            [round(reports[i]['accuracy'], 4) for i in range(len(model_names))],
            [round(reports[i]['weighted avg']['precision'], 4) for i in range(len(model_names))],
            [round(reports[i]['weighted avg']['recall'], 4) for i in range(len(model_names))],
            [round(reports[i]['weighted avg']['f1-score'], 4) for i in range(len(model_names))]
        ]

        fig_table = plt.figure(figsize=(8, 4))
        ax_table = fig_table.add_subplot(111)
        ax_table.axis('tight')
        ax_table.axis('off')

        row_name = ['Accuracy', 'Precision', 'Recall', 'F1 score']
        table = ax_table.table(cellText=values, colLabels=model_names, rowLabels=row_name, loc='center', cellLoc='center', colLoc='center', fontsize=20)
        table.auto_set_font_size(False)
        table.set_fontsize(10)

