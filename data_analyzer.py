import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer():
    def __init__(self, data):
        self.data = data
        self.headers = self.data.columns.tolist()

    def build_descriptive_data_table(self):
        values = []
        for col in self.headers:
            row = []
            desc = self.data[col].describe().round(2)

            row.append(col)
            row.append(desc['count'])
            row.append(desc['std']) # Standard deviation
            row.append(desc['min'])
            row.append(desc['25%'])
            row.append(desc['50%'])
            row.append(desc['75%'])
            row.append(desc['max'])

            values.append(row)

        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=values, colLabels=['', 'Count', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max'], loc='center', fontsize=20)
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        plt.show()

    def build_histogram_of_data_distribution(self):
        row_len = 4
        col_len = 5
        fig, ax = plt.subplots(row_len, col_len, tight_layout=True)

        i, j = 0, 0
        for col in self.headers:
            ax[i][j].hist(self.data[col], bins=10)
            ax[i][j].set_title(col)

            j += 1

            if (j >= col_len):
                i += 1
                j = 0

            if (i >= row_len):
                break

        plt.show()

    def build_correlation_heatmap(self):
        # Computes the correlation matrix
        correlation_matrix = self.data.corr(numeric_only=True)

        sns.heatmap(correlation_matrix, cmap="mako", annot=True)

        plt.show()
