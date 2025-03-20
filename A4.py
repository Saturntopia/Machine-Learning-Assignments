import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# Load dataset
df = pd.read_excel("../Data/RADCURE_Clinical_v04_20241219.xlsx")

# Define columns
time_col, event_col = "survival_time", "event"
covariates = ["Age", "Sex", "ECOG PS", "Smoking PY", "Smoking Status", "Ds Site", "Subsite"]

# Kaplan-Meier Analysis
def km_analysis(df, group_col):
    kmf = KaplanMeierFitter()
    for group in df[group_col].unique():
        mask = df[group_col] == group
        kmf.fit(df[mask][time_col], df[mask][event_col], label=str(group))
        kmf.plot()
    plt.title(f"Kaplan-Meier: {group_col}")
    plt.show()
    
    g1, g2 = df[group_col].unique()[:2]
    results = logrank_test(df[df[group_col] == g1][time_col], df[df[group_col] == g2][time_col], 
                           df[df[group_col] == g1][event_col], df[df[group_col] == g2][event_col])
    print(f"Log-rank test p-value: {results.p_value}")

# Cox Regression
def cox_regression(df):
    cph = CoxPHFitter()
    cph.fit(df[[time_col, event_col] + covariates], duration_col=time_col, event_col=event_col)
    cph.print_summary()
    cph.plot()
    plt.show()

# Random Survival Forest
def rsf_model(df):
    df = df.dropna()
    y = Surv.from_arrays(df[event_col].astype(bool), df[time_col])
    X = df[covariates]
    rsf = RandomSurvivalForest(n_estimators=100, random_state=42)
    rsf.fit(X, y)
    print(f"RSF C-index: {rsf.score(X, y):.4f}")
    pd.Series(rsf.feature_importances_, index=covariates).plot(kind='bar', title='Feature Importance')
    plt.show()

# Run analyses
km_analysis(df, "Smoking Status")
cox_regression(df)
rsf_model(df)
