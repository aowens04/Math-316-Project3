# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas==2.3.3",
#     "scikit-learn==1.7.2",
# ]
# ///

import marimo

__generated_with = "0.17.8"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return (
        ColumnTransformer,
        LinearRegression,
        OneHotEncoder,
        Pipeline,
        StandardScaler,
        mean_absolute_error,
        mean_squared_error,
        pd,
        r2_score,
        train_test_split,
    )


@app.cell
def _(pd):
    df = pd.read_csv("bank-full.csv", sep=";")

    #Change to numeric so it can predict
    #0=no 1=yes
    df["y_numeric"] = df["y"].map({"yes": 1, "no": 0})
    return (df,)


@app.cell
def _(df):
    #Define features and target

    numeric_features = ["age", "duration", "campaign", "previous"]
    categorical_features = ["job", "marital", "education", "default", 
                            "housing", "loan", "contact", "poutcome"]

    X = df[numeric_features + categorical_features]
    y = df["y_numeric"]
    return X, categorical_features, numeric_features, y


@app.cell
def _(mean_absolute_error, mean_squared_error, r2_score):
    #Create error metrics function

    def print_error_metrics(y_true, y_pred, model_name="Model"):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{model_name} Performance:")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R^2 Score: {r2:.4f}")
    return (print_error_metrics,)


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(
    ColumnTransformer,
    LinearRegression,
    OneHotEncoder,
    Pipeline,
    StandardScaler,
    X_test,
    X_train,
    categorical_features,
    numeric_features,
    print_error_metrics,
    y_test,
    y_train,
):
    #Build model
    def build_model_pipeline():
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric_features),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression())
            ]
        )
        return model_pipeline

    #Fit model

    model_pipeline = build_model_pipeline()
    model_pipeline.fit(X_train, y_train)

    #Predict
    y_pred = model_pipeline.predict(X_test)

    #Evaluate
    print_error_metrics(y_test, y_pred)
    return


if __name__ == "__main__":
    app.run()
