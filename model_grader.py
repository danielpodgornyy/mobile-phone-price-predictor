# Imports for running models
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as NB

# Imports for plotting
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class ModelGrader():
    def __init__(self, data):
        self.data = data

    def predict_prices(self):
        x = self.data.drop(columns=['price_range'])
        y = self.data['price_range']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        fig, ax = plt.subplots(2, 2)

        models = [SVM(), DT(), KNN(), NB()]
        model_names = ['SVM', 'DT', 'KNN', 'NB']
        reports = []
        accuracies = []

        i, j = 0, 0
        for model, model_name in zip(models, model_names):
            y_pred = model.fit(x_train, y_train).predict(x_test)

            self.plot_model(model_name, i, j, y_test, y_pred, ax)

            report = classification_report(y_test, y_pred, output_dict=True, target_names=['0', '1', '2', '3'])
            reports.append(report)

            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            j += 1
            if j >= 2:
                i += 1
                j = 0

            if i >= 2:
                break

        self.add_table(model_names, reports, accuracies)

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

    def add_table(self, model_names, reports, accuracies):
        row_names = [''

