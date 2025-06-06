import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

import tools


def main():
    # Initialize
    tool = tools.Utils()
    tool.set_seed(seed=2025)
    tool.init_logger()
    method = xgb.XGBClassifier
    imputation_flag = False
    imputation_policy = "remove"

    # -------------------------------------------------------------------------------
    # Data Exploration
    # -------------------------------------------------------------------------------
    tool.preprocess(data_filename="data/churn_data_2025.xlsx", sheet_name="E Comm")
    preprocess_data = tool.get_preprocess_data()

    # Perform EDA
    tool.eda_plots(preprocess_data)

    # -------------------------------------------------------------------------------
    # Analysis from CX perspectives
    # -------------------------------------------------------------------------------
    tool.insights(preprocess_data)
    df = tool.get_prediction_data()
    tool.split(df, target_column="Churn", test_size=0.3)
    split_data = tool.get_split_data()
    tool.features_explainer(split_data)

    # -------------------------------------------------------------------------------
    # Insights and Recommendations
    # -------------------------------------------------------------------------------
    split_data = None
    if method == RandomForestClassifier:
        tool.one_hot_enconding(preprocess_data)
        df = tool.get_numerical_data()

        # X_train, X_test, y_train, y_test
        tool.split(df, target_column="Churn", test_size=0.3)
        split_data = tool.get_split_data()

        # Impute data (train and test separately)
        if imputation_flag:
            tool.imputation(df=split_data[0], target_df=split_data[2], imputation_policy=imputation_policy)
            split_data[0] = tool.get_imputation_data()
            split_data[2] = tool.imputation_target()

            tool.imputation(df=split_data[1], target_df=split_data[3], imputation_policy=imputation_policy)
            split_data[1] = tool.get_imputation_data()
            split_data[3] = tool.imputation_target()
    elif method == xgb.XGBClassifier:
        df = tool.get_prediction_data()
        tool.split(df, target_column="Churn", test_size=0.3)
        split_data = tool.get_split_data()

    # Churn Prediction and Probability
    tool.predict_churn(split_data, method=method)


if __name__ == "__main__":
    main()
