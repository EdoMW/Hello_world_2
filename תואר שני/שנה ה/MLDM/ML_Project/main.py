import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.legend_handler import HandlerBase
import re
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import GridSearchCV, train_test_split
from time import localtime, strftime, time

pd.set_option('display.max_columns', None)
pd.set_option('max_columns', None)


def print_time_line_sep(msg):
    """
    :param msg: massage to print
    print len sep with current date time.
    """
    dt_fmt = "%d/%m/%Y %H:%M:%S"
    line_msg = " " * 41 + msg
    line_sep = "-" * 35 + " " + strftime(dt_fmt, localtime()) + " " + "-" * 35 + '\n'
    print(line_msg)
    print(line_sep)


def read_data():
    data = r'C:\Users\Administrator\Desktop\Edo_dir\MSc\5 year\MLDM\project_MLDM\data.csv'
    df = pd.read_csv(data)
    return df


def missing_values(df):
    """ Missing values """
    print_time_line_sep("Missing values")
    print(df.isnull().sum())
    print(df.isnull().mean().round(4))  # It returns percentage of missing values in each column in the dataframe.

    # assert that there are no missing values in the dataframe
    # assert pd.notnull(df).all().all(), "missing values exits!"


def fix_value_spaces_and_names(df):
    df = df.replace(to_replace=[" <=50K.", " <=50K"], value="<=50K")
    df = df.replace(to_replace=[" >50K", " >50K."], value=">50K")
    df = df.replace(to_replace=[" ?", "?"], value=None)
    df = df.replace(to_replace=[np.nan], value=None)
    print(df.columns)
    cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    df[cols] = df[cols].apply(lambda x: x.str.strip())
    # df[cols].apply(lambda x: print(pd.unique(df['workclass']).tolist()))
    # print(df.isin([' ? ']).any())
    # column_names = df.columns.tolist()
    # print(type(df['income'].iloc[0:1]))
    # for i in range(len(column_names)):
    #     print("%%%%", i, type(df[column_names[i]].iloc[0:1].dtypes))
    #     df[df.columns] = df.apply(lambda x: x.str.strip())
    #     # if type(df[column_names[i]].iloc[0:1]) is object:
    #     #     print(i, ": ", df[column_names[i]])
    #     #     df[column_names[i]] = df[column_names[i]].strip()
    #     # else:
    #     #     print(i, df[column_names[i]].iloc[0:1])

    return df


def check_columns():
    column_names = df.columns.tolist()
    for i in range(len(column_names)):
        print(column_names[i])
        print(df[column_names[i]].isnull().sum())
        print(df[column_names[i]].nunique())
        print(df[column_names[i]].value_counts())
        if type(df[column_names[i]]) is str:
            print(df[column_names[i]].str.startswith(' '), '\n')


def income_general_distribution():
    print_time_line_sep("Descriptive Statistics")
    # # visualize frequency distribution of income variable
    f = plt.subplots(1, 1, figsize=(10, 8))
    # ax[0] = df['income'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
    df['income'].value_counts().plot(kind='pie',   autopct='%1.1f%%', colors=["red", "green"],
            startangle=90, shadow=False,  legend=True, fontsize=19, labels=["<=50K$",">50K$"])
    plt.title('Income distribution', fontsize=22, fontweight="bold")
    plt.legend(fontsize='x-large')
    plt.ylabel('', fontsize=20)
    plt.show()


def distribution_per_variable(df, df_names):
    print_time_line_sep("Descriptive Statistics")
    for i in df_names:
        # visualize frequency distribution of variable
        f = plt.subplots(1, 1, figsize=(10, 8))
        # ax[0] = df['income'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
        a = sns.displot(x=df[i], data=df, palette="Set1")
        plt.title(f'{i} distribution', fontsize=22, fontweight="bold")
        # plt.legend(fontsize='x-large')
        # plt.ylabel('', fontsize=20)
        plt.show()


def age_distribution(df):
    # # visualize frequency distribution of income variable
    f = plt.subplots(1, 1, figsize=(11, 8))
    g = sns.displot(df, x=df['age'], bins=20)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('Age distribution', fontsize=14, fontweight="bold")
    plt.show()


def workclass_distribution(df):
    df['workclass'].hist()
    plt.show()

    # class TextHandler(HandlerBase):
    #     def create_artists(self, legend, tup, xdescent, ydescent,
    #                        width, height, fontsize, trans):
    #         tx = Text(width / 2., height / 2, tup[0], fontsize=fontsize,
    #                   ha="center", va="center", color=tup[1], fontweight="bold")
    #         return [tx]
    #
    # # # visualize frequency distribution of income variable
    # f = plt.subplots(1, 1, figsize=(11, 8))
    # g = sns.countplot(data=df, x=df['workclass'])
    # handltext = ["SG", "SENI", "P", "F", "LG", "SEI", "WP", "NW"]
    # labels = pd.unique(df['workclass']).tolist()
    # t = g.get_xticklabels()
    # labeldic = dict(zip(handltext, labels))
    # labels = [labeldic[h.get_text()] for h in t]
    # handles = [(h.get_text(), c.get_fc()) for h, c in zip(t, g.patches)]
    #
    # g.legend(handles, labels, handler_map={tuple: TextHandler()})
    #
    # g.fig.subplots_adjust(top=.95)
    # plt.legend(loc='center right', title='Work class')
    # g.ax.set_title('Workclass distribution', fontsize=14, fontweight="bold")
    # plt.show()


def plot_descriptive_statistics(df):
    income_general_distribution()


    # # Visualize income wrt race # uncomment to plot separately
    # f, ax = plt.subplots(figsize=(10, 8))
    # ax[0] = sns.countplot(x="income", hue="race", data=df, palette="Set1") #
    # ax[0].set_title("Frequency distribution of income variable wrt race")
    # # # ax[1] = sns.countplot(x="income", hue="race", data=df, palette="Set1") #
    # # # ax[1].set_title("Frequency distribution of income variable wrt race")
    # plt.show()


def describe_df(df):
    print_time_line_sep("describe data")
    print(df.describe(include=['object']).T)
    print(df.describe().T)


def corr_matrices(df):
    corr = df.corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Spearman")
    plt.show()
    corr = df.corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Pearson")
    plt.show()

def tune_pipeline(pipeline, hyper_parameters_grid, X, y, n_folds=5, n_jobs=1):
    """Tune the pipeline using exhaustive grid search over specified parameter values for an estimator.
    If hyper_ + HOG pipeline
    hyper_parameters_grid: Dict.
        dictionary with the hyper-parameters space to search.
        HOG parameters should have keys starting with HOG__
        SVM parameters should have keys starting with SVM__
        ImageScaler parameters should have keys starting with scale__

    X: array.
        training data array of shape (N, img_scaled_size, img_scaled_size)
    y: array.
        training data labels of shape N
    n_folds: intparameters_grid contains one value per parameter this is equivalent to just fitting the pipeline with the
    selected configurations as refit is set to true
    (Refit an estimator using the best found parameters on the whole dataset).

    Parameters
    ----------
    pipeline: Pipeline.
        an SVM, optional (default=5).
        number of folds for cross validation process
    n_jobs: int, optional (default=1)
        number of jobs to run in parallel
        -1 means use all processors

    Returns
    -------
    search: GridSearchCV
        fitted GridSearchCV object

    Notes
    -------
    This function is generic and could be used with other pipelines. The explanation is suited for this exercise

    See Also
    ------
    get_hog_svm_pipeline
    sklearn.model_selection.GridSearchCV
    """
    search = GridSearchCV(
        pipeline,
        hyper_parameters_grid,
        cv=n_folds,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
        scoring='accuracy'
    )
    search.fit(X, y)
    return search


if __name__ == '__main__':
    df = read_data()
    names = df.columns.tolist()
    print('The shape of the dataset : ', df.shape)
    # df.info()
    # describe_df(df)
    df = fix_value_spaces_and_names(df)
    missing_values(df)

    # print(df.describe(include='all').T)
    # target value
    # check_columns()
    # print(df['income'].isnull().sum())
    # print(df['income'].nunique())
    # print(df['income'].value_counts())
    # print(df["native.country"].str.startswith(' '))
    # corr_matrices(df)
    # stat, p, dof, expected = chi2_contingency(df['occupation'], df['workclass'])


    # plt.matshow(df.corr())
    # plt.show()

    # plot_descriptive_statistics(df)
    # age_distribution(df)

    workclass_distribution(df)
    # distribution_per_variable(df,names)

    # X, y = df.iloc[:,:14], df.iloc[:, 14:15]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # tune_pipeline(pipeline, hyper_parameters_grid, X, y, n_folds=5, n_jobs=1)