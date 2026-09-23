import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score, confusion_matrix,
                              classification_report)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("tour_package_cleaned.csv")
print(f"Loaded: {df.shape}")

fe = df.copy()
desig_rank = {"Executive":1,"Manager":2,"Senior Manager":3,"AVP":4,"VP":5}
fe["DesignationRank"]  = fe["Designation"].map(desig_rank)
fe["IncomePerPerson"]  = fe["MonthlyIncome"] / fe["NumberOfPersonVisiting"]
fe["EngagementScore"]  = (fe["NumberOfFollowups"] + fe["PitchSatisfactionScore"]
                           + fe["DurationOfPitch"] / fe["DurationOfPitch"].max() * 5).round(3)
fe["TravelPropensity"] = (fe["NumberOfTrips"] + fe["Passport"]*3 + fe["NumberOfChildrenVisiting"]).round(3)
fe["FamilySize"]       = fe["NumberOfPersonVisiting"] + fe["NumberOfChildrenVisiting"]
fe["IsMarried"]        = (fe["MaritalStatus"] == "Married").astype(int)
fe["AgeGroup"]         = pd.cut(fe["Age"], bins=[0,25,35,45,100],
                                 labels=["Young","Early-Mid","Mid","Senior"]).astype(str)
fe["IncomeTier"]       = pd.cut(fe["MonthlyIncome"], bins=[0,18000,23000,30000,1e9],
                                 labels=["Low","Mid","High","Premium"]).astype(str)
fe["HighValue"]        = ((fe["MonthlyIncome"] > fe["MonthlyIncome"].median()) &
                           (fe["Passport"] == 1)).astype(int)
fe["CompanyInvited"]   = (fe["TypeofContact"] == "Company Invited").astype(int)
new_feats = ["DesignationRank","IncomePerPerson","EngagementScore","TravelPropensity",
             "FamilySize","IsMarried","AgeGroup","IncomeTier","HighValue","CompanyInvited"]

TARGET   = "ProdTaken"
fe_model = fe.drop(columns=["CustomerID"])
le_dict  = {}
for col in fe_model.select_dtypes("object").columns:
    le = LabelEncoder()
    fe_model[col] = le.fit_transform(fe_model[col].astype(str))
    le_dict[col] = le

X = fe_model.drop(columns=[TARGET])
y = fe_model[TARGET]
feature_names = X.columns.tolist()
scaler  = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)
print(f"Train: {X_train_sm.shape}  Test: {X_test.shape}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
    "XGBoost":             xgb.XGBClassifier(n_estimators=200, random_state=42,
                                              eval_metric="logloss", verbosity=0),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cv_s    = cross_val_score(model, X_train_sm, y_train_sm, cv=cv, scoring="roc_auc", n_jobs=-1)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    results[name] = {
        "auc":    round(roc_auc_score(y_test, y_proba), 4),
        "ap":     round(average_precision_score(y_test, y_proba), 4),
        "cv_auc": round(cv_s.mean(), 4),
        "cv_std": round(cv_s.std(), 4),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "prec": prec.tolist(), "rec": rec.tolist(),
        "cm":     confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }
    print(f"  {name:<25} AUC={results[name]['auc']}")

fi_rf  = pd.Series(models["Random Forest"].feature_importances_, index=feature_names).sort_values(ascending=False)
fi_xgb = pd.Series(models["XGBoost"].feature_importances_,       index=feature_names).sort_values(ascending=False)

print("Computing SHAP...")
explainer = shap.TreeExplainer(models["XGBoost"])
shap_vals = explainer.shap_values(X_test)

with open("artifacts/pipeline.pkl", "wb") as f:
    pickle.dump({
        "feature_names": feature_names,
        "results":       results,
        "best_model":    models["XGBoost"],
        "scaler":        scaler,
        "le_dict":       le_dict,
        "fi_rf":         fi_rf.to_dict(),
        "fi_xgb":        fi_xgb.to_dict(),
        "shap_vals":     shap_vals.tolist(),
        "X_test":        X_test.to_dict(orient="list"),
        "y_test":        y_test.tolist(),
        "fe_df":         fe.to_dict(orient="list"),
        "new_feats":     new_feats,
    }, f)

print("Done — artifacts/pipeline.pkl saved.")
