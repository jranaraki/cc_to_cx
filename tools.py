import io
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight


def __store_console__(func, filename, skip_head, skip_tail):
    """
    Store console output to a CSV file
    :param func: The function that generates console output
    :param filename: Filename to be stored the results in
    :param skip_head: Number of lines to be skipped from the head
    :param skip_tail: Number of lines to be skipped from the tail
    :return:
    """
    buffer = io.StringIO()
    func(buf=buffer)
    info_output = buffer.getvalue()
    lines = info_output.splitlines()[skip_head:-skip_tail]
    data = pd.DataFrame([line.split() for line in lines[2:]], columns=lines[0].split())
    data.to_csv(filename, index=False)


class Utils:
    """
    Utils class that include all the functionalities required to work on the input data
    """
    def __init__(self):
        self.preprocess_data = None
        self.log_filename = None
        self.numerical_data = None
        self.seed = None
        self.class_counts = None
        self.prediction_data = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.split_data = None
        self.imputation_data = None
        self.imputation_target = None
        self.logger = None
        self.weights = None
        self.logs_path = "logs"
        self.plots_path = "plots"
        self.analysis_path = "analysis"
        self.models_path = "models"
        os.makedirs("logs", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("analysis", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        self.init_logger()


    def init_logger(self):
        """
        Initializing the logger and output file for it
        :return:
        """
        self.log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%s") + ".txt"
        log_path = os.path.join(self.logs_path, self.log_filename)

        self.logger = logging.getLogger(__name__)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        logging.basicConfig(level=logging.INFO, filename=log_path, format='%(asctime)s - %(levelname)s - %(message)s')


    def set_seed(self, seed):
        """
        Setter for the random seed for reproducibility
        :param seed: Input seed
        :return:
        """
        self.seed = seed
        np.random.seed(self.seed)


    def get_preprocess_data(self):
        """
        Getter for preprocess data dataframe
        :return: preprocessed dataframe
        """
        return self.preprocess_data


    def get_split_data(self):
        """
        Getter for split data array
        :return: An array of X_train, X_test, y_train, y_test
        """
        return self.split_data


    def get_imputation_data(self):
        """
        Getter for imputed features
        :return: Either imputed features or the data with rows with missing values removed
        """
        return self.imputation_data


    def get_imputation_target(self):
        """
        Getter for the target after applying imputation
        :return: Usually the original target unless rows with missing values are removed
        """
        return self.imputation_target


    def get_numerical_columns(self):
        """
        Getter for the data with only numerical columns
        :return: Data columns that are only numerical
        """
        return self.numerical_columns


    def get_numerical_data(self):
        """
        Getter for all numerical data
        :return: The original data with one-hot-encoding feature for the categorical ones
        """
        return self.numerical_data


    def get_prediction_data(self):
        """
        Getter for prediction data
        :return: The original data with categorical columns instead of object
        """
        self.preprocess_data[self.categorical_columns] = self.preprocess_data[self.categorical_columns].astype("category")
        return self.preprocess_data


    def preprocess(self, data_filename, sheet_name):
        """
        Read in input data file
        :param data_filename: Data filename
        :param sheet_name: Sheet name to be read
        :return:
        """
        try:
            # Read in raw data
            self.logger.info(f"Reading in \"{sheet_name}\" sheet from \"{data_filename}\"...")
            self.preprocess_data = pd.read_excel(data_filename, sheet_name=sheet_name)
            self.preprocess_data = self.preprocess_data.set_index("CustomerID")

            # Merging similar categories
            self.preprocess_data["PreferredLoginDevice"] = self.preprocess_data["PreferredLoginDevice"].replace("Phone", "Mobile Phone")
            self.preprocess_data["PaymentMode"] = self.preprocess_data["PaymentMode"].replace("CC", "Credit Card")
            self.preprocess_data["PaymentMode"] = self.preprocess_data["PaymentMode"].replace("COD", "Cash on Delivery")
            self.preprocess_data["OrderCat"] = self.preprocess_data["OrderCat"].replace("Mobile", "Mobile Phone")

            __store_console__(self.preprocess_data.info, os.path.join(self.analysis_path, "data_info.csv"), 3, 2)

            self.logger.info(f"Number of rows and columns: {self.preprocess_data.shape}")
            self.class_counts = self.preprocess_data["Churn"].value_counts().to_dict()
            self.logger.info(f"Class counts: {self.class_counts}")

            self.numerical_columns = self.preprocess_data.select_dtypes(include='number').columns.to_list()
            self.logger.info(f"Numerical columns: {self.numerical_columns}")

            self.categorical_columns = self.preprocess_data.select_dtypes(exclude='number').columns.to_list()
            self.logger.info(f"Categorical columns: {self.categorical_columns}")
        except ValueError:
            self.logger.error(f"\"{sheet_name}\" does not exist!")
        except FileNotFoundError:
            self.logger.error(f"{data_filename} is missing!")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")


    def to_categorical(self, df):
        """
        Convert string/object columns to categorical
        :param df: Input dataframe
        :return:
        """
        df.loc[:, self.categorical_columns] = df.loc[:, self.categorical_columns].astype("category")
        self.prediction_data = df


    def one_hot_enconding(self, df):
        """
        One-hot encoding of categorical columns
        :param df: Input dataframe
        :return:
        """
        self.numerical_data = pd.get_dummies(df, columns=self.categorical_columns, drop_first=True).astype(int, errors='ignore')


    def eda_plots(self, df):
        """
        Generate EDA plots
        :param df: Input dataframe
        :return:
        """
        plot_html = []
        for col in df.columns:
            if col in self.categorical_columns:
                fig = px.pie(df, names=col, title=f"Distribution of {col}")
                plot_html.append(fig.to_html(full_html=False, include_plotlyjs=False))
            elif col in self.numerical_columns:
                fig = px.histogram(df, x=col, nbins=20, title=f"{col} Distribution")
                plot_html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        corr = df[self.numerical_columns].corr().abs()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='Viridis',
            zmin=0, zmax=1,
            colorbar=dict(title="Correlation")
        ))
        fig.update_layout(title='Correlation Heatmap')
        plot_html.append(fig.to_html(full_html=False, include_plotlyjs=False))

        # Save all plots to one HTML file
        html_file = os.path.join(self.plots_path, "eda_dashboard.html")
        with open(html_file, 'w') as f:
            f.write("<html><head><title>EDA Dashboard</title></head><body>\n")
            f.write("<h1>EDA Plots</h1>\n")
            for html in plot_html:
                f.write(html)
                f.write("<hr>\n")
            f.write("</body></html>")


    def __sample_weights__(self, df):
        """
        Calculate sample weights for XGBoost to handle imbalanced dataset
        :param df: Input dataframe
        :return:
        """
        self.weights = compute_sample_weight("balanced", df)


    def features_explainer(self, dfs):
        """
        Calculate features explainer using SHAP
        :param dfs: An array of X_train, X_test, y_train, y_test
        :return:
        """
        X_train, _, y_train, _ = dfs
        model = xgb.XGBClassifier(enable_categorical=True, random_state=self.seed)
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_train)
        fig = shap.plots.beeswarm(explanation, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, "shap_values.png"), dpi=300)


    def split(self, df, target_column="Churn", test_size=0.3):
        """
        Split data into train and test dataset
        :param df: Input dataframe
        :param target_column: Target column name
        :param test_size: Test size
        :return:
        """
        if df is None:
            self.logger.error(f"No dataframe is found!")
        else:
            self.split_data = train_test_split(df.loc[:,df.columns!=target_column], df[target_column], test_size=test_size, random_state=self.seed, stratify=df[target_column])


    def imputation(self, df, target_df, imputation_policy):
        """
        Impute the data by replacing/removing missing items
        :param df: Input dataframe
        :param target_df: Target dataframe
        :param imputation_policy:
        :return:
        """
        if df is None:
            self.logger.error(f"No dataframe is found!")
        else:
            self.imputation_data = df
            self.imputation_target = target_df
            missing_rows = self.imputation_data[self.imputation_data.isna().any(axis=1)]
            if imputation_policy == "impute":
                if missing_rows is not None:
                    self.logger.info(f"Found {len(missing_rows)} rows with missing values.")
                    for col in self.imputation_data.columns:
                        if pd.isna(self.imputation_data[col]).any():
                            if (self.imputation_data[col].dtype == "object") or (self.imputation_data[col].dtype == "string") or (self.imputation_data[col].dtype == "categorical"):
                                categorical_imputer = SimpleImputer(strategy="most_frequent")
                                self.imputation_data[[col]] = categorical_imputer.fit(self.imputation_data[[col]])
                            else:
                                numerical_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                                self.imputation_data[[col]] = numerical_imputer.fit(self.imputation_data[[col]])
            elif imputation_policy == "remove":
                self.imputation_data = self.imputation_data[~self.imputation_data.index.isin(missing_rows.index)]
                self.imputation_target = self.imputation_target[~self.imputation_target.index.isin(missing_rows.index)]
            else:
                self.logger.error("Unexpected imputation_policy!")


    def predict_churn(self, dfs, method=RandomForestClassifier):
        """
        Churn prediction and probability calculations and store the trianed model
        :param dfs: An array of X_train, X_test, y_train, y_test
        :param method: Classification method (RandomForestClassifier or xgb.XGBClassifier)
        :return:
        """
        X_train, X_test, y_train, y_test = dfs
        if method == RandomForestClassifier:
            model = method(class_weight="balanced", random_state=self.seed)
            model.fit(X_train, y_train)
        elif method == xgb.XGBClassifier:
            self.__sample_weights__(y_train)
            model = method(enable_categorical=True, random_state=self.seed)
            model.fit(X_train, y_train, sample_weight=self.weights)
        y_pred = model.predict(X_test)
        prob = model.predict_proba(X_test)

        models_path = os.path.join(self.models_path, self.log_filename[:-4]+ ".pkl")
        with open(models_path, 'wb') as model_file:
            pickle.dump(model, model_file)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel().tolist()
        self.logger.info(f"TN, FP, FN, TP: {tn}, {fp}, {fn}, {tp}")
        self.logger.info(f"Classification method: {method.__name__}")
        self.logger.info(f"Classification accuracy: {accuracy_score(y_test, y_pred)}")
        self.logger.info(f"f1-score: {f1_score(y_test, y_pred)}")

        churn_df = X_test.reset_index(drop=False)[["CustomerID"]]
        churn_df.loc[:, "Probability"] = prob[:, 1]
        churn_df = churn_df.sort_values(by="Probability", ascending=False)
        churn_df.to_csv(os.path.join(self.analysis_path, f"churn_probability_{method.__name__}.csv"), index=False)