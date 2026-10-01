"""EDA, charts and model comparison for the Student Performance Predictor.
Run from the project root:  python analysis/eda_and_models.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid")
df = pd.read_csv("notebook/data/stud.csv")
df.columns = df.columns.str.replace(" ", "_").str.replace("/", "_")
df["average_score"] = df[["math_score", "reading_score", "writing_score"]].mean(axis=1)
df["math_pass"] = np.where(df["math_score"] >= 40, "Pass", "Fail")

# ---------- Data quality ----------
print("Shape:", df.shape, "| Nulls:", df.isnull().sum().sum(), "| Duplicates:", df.duplicated().sum())

# ---------- Individual charts ----------
cats = {
    "test_preparation_course": "Test prep course",
    "lunch": "Lunch type",
    "parental_level_of_education": "Parental education",
    "race_ethnicity": "Group",
    "gender": "Gender",
}
for col, title in cats.items():
    order = df.groupby(col)["math_score"].mean().sort_values().index
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=df, x=col, y="math_score", order=order, errorbar=None, color="#4C72B0")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f")
    plt.title(f"Average math score by {title}")
    plt.xlabel(title); plt.ylabel("Average math score")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(f"images/math_by_{col}.png", dpi=130); plt.close()

plt.figure(figsize=(5.5, 4.5))
sns.heatmap(df[["math_score", "reading_score", "writing_score"]].corr(), annot=True, cmap="Blues", vmin=0.7, vmax=1)
plt.title("Correlation between scores"); plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=130); plt.close()

# ---------- One-page dashboard image ----------
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle("Student Performance Dashboard (1,000 students)", fontsize=18, fontweight="bold")
kpis = [("Students", f"{len(df):,}"), ("Avg math", f"{df.math_score.mean():.1f}"),
        ("Avg reading", f"{df.reading_score.mean():.1f}"), ("Avg writing", f"{df.writing_score.mean():.1f}"),
        ("Math pass % (>=40)", f"{(df.math_score >= 40).mean() * 100:.0f}%")]
fig.text(0.5, 0.925, "     |     ".join(f"{k}: {v}" for k, v in kpis), ha="center", fontsize=12)
for ax, (col, title) in zip(axes.flat[:4], list(cats.items())[:4]):
    order = df.groupby(col)["math_score"].mean().sort_values().index
    sns.barplot(data=df, x=col, y="math_score", order=order, errorbar=None, color="#4C72B0", ax=ax)
    for c in ax.containers: ax.bar_label(c, fmt="%.1f", fontsize=9)
    ax.set_title(f"Math score by {title}"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=25)
sns.scatterplot(data=df, x="reading_score", y="math_score", hue="test_preparation_course", alpha=.6, ax=axes[1, 1])
axes[1, 1].set_title("Reading vs Math score")
sns.histplot(df["math_score"], bins=20, kde=True, color="#55A868", ax=axes[1, 2])
axes[1, 2].set_title("Math score distribution")
plt.tight_layout(rect=[0, 0, 1, 0.91]); plt.savefig("images/dashboard.png", dpi=110); plt.close()

# ---------- Model comparison ----------
X, y = df.drop(columns=["math_score", "average_score", "math_pass"]), df["math_score"]
num = ["reading_score", "writing_score"]; cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(), cat)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso(alpha=0.1),
          "Random Forest": RandomForestRegressor(random_state=42),
          "Gradient Boosting": GradientBoostingRegressor(random_state=42)}
rows, fitted = [], {}
for name, m in models.items():
    p = Pipeline([("pre", pre), ("m", m)]).fit(X_tr, y_tr); fitted[name] = p
    pred = p.predict(X_te)
    rows.append({"Model": name, "R2": r2_score(y_te, pred), "MAE": mean_absolute_error(y_te, pred),
                 "RMSE": np.sqrt(mean_squared_error(y_te, pred))})
res = pd.DataFrame(rows).sort_values("R2", ascending=False).round(4)
res.to_csv("analysis/model_comparison.csv", index=False); print(res.to_string(index=False))

best = fitted[res.iloc[0]["Model"]]
imp = permutation_importance(best, X_te, y_te, n_repeats=20, random_state=42, scoring="r2")
fi = pd.Series(imp.importances_mean, index=X.columns).sort_values()
fi.to_csv("analysis/feature_importance.csv", header=["importance"]); print(fi.round(4))
plt.figure(figsize=(7, 4)); fi.plot.barh(color="#C44E52")
plt.title(f"Feature importance ({res.iloc[0]['Model']})"); plt.xlabel("Drop in R2 when shuffled")
plt.tight_layout(); plt.savefig("images/feature_importance.png", dpi=130); plt.close()
