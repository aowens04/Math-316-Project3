# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.0.0",
#     "marimo",
#     "matplotlib==3.10.7",
#     "pandas==2.3.3",
#     "scikit-learn==1.7.2",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import pandas as pd
    import altair as alt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        ConfusionMatrixDisplay,
        classification_report,
        roc_curve, 
        roc_auc_score
    )
    return (
        ColumnTransformer,
        LogisticRegression,
        OneHotEncoder,
        Pipeline,
        StandardScaler,
        accuracy_score,
        alt,
        classification_report,
        confusion_matrix,
        f1_score,
        pd,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        train_test_split,
    )


@app.cell
def _(
    ColumnTransformer,
    LogisticRegression,
    OneHotEncoder,
    Pipeline,
    StandardScaler,
    pd,
    train_test_split,
):


    #Model 1

    df = pd.read_csv("bank-full.csv", sep=";")

    target_names = ["yes", "no"]
    numeric_features = ["age", "balance", "campaign", "previous"]
    categorical_features = [
        "job", "marital", "education", "default", "housing", 
        "loan", "contact", "poutcome"
    ]

    X = df[numeric_features + categorical_features]
    y = df["y"]


    #train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    #logistic regression
    def build_logistic_pipeline():
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric_features),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        model = LogisticRegression(max_iter=500, class_weight="balanced")

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model)
            ]
        )


    #fit model
    logreg_model = build_logistic_pipeline()
    logreg_model.fit(X_train, y_train)

    #predict probabilities
    y_pred_prob = logreg_model.predict_proba(X_test)[:, 1]
    y_pred = logreg_model.predict(X_test)
    return (
        categorical_features,
        logreg_model,
        numeric_features,
        target_names,
        y_pred,
        y_pred_prob,
        y_test,
    )


@app.cell
def _(
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    target_names,
    y_pred,
    y_test,
):
    #Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="yes")
    recall = recall_score(y_test, y_pred, pos_label="yes")
    f1 = f1_score(y_test, y_pred, pos_label="yes")

    # Print metrics
    print("=" * 50)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=target_names))
    return


@app.cell
def _():
    return


@app.cell
def _(
    alt,
    categorical_features,
    confusion_matrix,
    logreg_model,
    numeric_features,
    pd,
    roc_auc_score,
    roc_curve,
    y_pred,
    y_pred_prob,
    y_test,
):
    ohe = logreg_model.named_steps["preprocessor"].named_transformers_["categorical"]
    encoded_cat_names = ohe.get_feature_names_out(categorical_features)

    all_feature_names = numeric_features + list(encoded_cat_names)
    coefficients = logreg_model.named_steps["classifier"].coef_[0]

    coef_df = (
        pd.DataFrame({"feature": all_feature_names, "coef": coefficients})
          .assign(abs_coef=lambda d: abs(d["coef"]))
          .sort_values("abs_coef", ascending=False)
    )

    feature_chart = (
        alt.Chart(coef_df.head(20))   # top 20 for readability
        .mark_bar()
        .encode(
            x=alt.X("abs_coef:Q", title="|Coefficient|"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            color=alt.condition(
                alt.datum.coef > 0,
                alt.value("#1b9e77"),
                alt.value("#d95f02")
            )
        )
        .properties(width=300, height=400, title="Top 20 Most Important Features")
    )

    # =====================================================
    # 2. PROBABILITY DISTRIBUTION PANEL
    # =====================================================
    prob_df = pd.DataFrame({"prob": y_pred_prob, "true": y_test.values})

    prob_chart = (
        alt.Chart(prob_df)
        .mark_bar()
        .encode(
            x=alt.X("prob:Q", bin=alt.Bin(step=0.05), title="Predicted P(y=yes)"),
            y="count()",
            color=alt.Color("true:N", title="True Outcome")
        )
        .properties(width=300, height=200, title="Predicted Probability Distribution")
    )

    # =====================================================
    # 3. CONFUSION MATRIX PANEL
    # =====================================================
    cm = confusion_matrix(y_test, y_pred, labels=["no", "yes"])

    cm_df = (
        pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Pred No", "Pred Yes"])
          .reset_index()
          .melt(id_vars="index", var_name="Predicted", value_name="Count")
          .rename(columns={"index": "Actual"})
    )

    cm_chart = (
        alt.Chart(cm_df)
        .mark_rect()
        .encode(
            x="Predicted:N",
            y="Actual:N",
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="reds")),
            tooltip=["Actual", "Predicted", "Count"]
        )
        .properties(width=200, height=200, title="Confusion Matrix")
    )

    # =====================================================
    # 4. ROC CURVE PANEL
    # =====================================================
    fpr, tpr, _ = roc_curve((y_test == "yes").astype(int), y_pred_prob)
    auc = roc_auc_score((y_test == "yes").astype(int), y_pred_prob)

    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})

    roc_chart = (
        alt.Chart(roc_df)
        .mark_line()
        .encode(
            x=alt.X("FPR:Q", title="False Positive Rate"),
            y=alt.Y("TPR:Q", title="True Positive Rate")
        )
        .properties(width=300, height=200, title=f"ROC Curve (AUC = {auc:.3f})")
    )

    # =====================================================
    # 5. COMBINE INTO MULTI-PANEL DASHBOARD (HORIZONTAL)
    # =====================================================
    dashboard = feature_chart | (prob_chart & cm_chart & roc_chart)

    dashboard
    return


if __name__ == "__main__":
    app.run()
