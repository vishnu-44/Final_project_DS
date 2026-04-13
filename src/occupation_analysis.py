import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


JOB_ZONE_LABELS = {
    1: "Zone 1: Little preparation",
    2: "Zone 2: Some preparation",
    3: "Zone 3: Medium preparation",
    4: "Zone 4: Considerable preparation",
    5: "Zone 5: Extensive preparation",
}


def load_sources(base_dir):
    job = pd.read_csv("{}/job_exposure.csv".format(base_dir))
    wage = pd.read_csv("{}/wage_data.csv".format(base_dir))
    soc = pd.read_csv("{}/SOC_Structure.csv".format(base_dir))
    return job, wage, soc


def build_occupation_dataset(base_dir):
    job, wage, soc = load_sources(base_dir)

    wage["soc_base"] = wage["SOCcode"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    canonical = wage[wage["SOCcode"].astype(str).str.endswith(".00")].copy()
    canonical["major_group_code"] = canonical["soc_base"].str.slice(0, 2) + "-0000"

    major = soc[soc["Major Group"].notna()][
        ["Major Group", "SOC or O*NET-SOC 2019 Title"]
    ].copy()
    major.columns = ["major_group_code", "major_group_title"]

    merged = job.merge(canonical, left_on="occ_code", right_on="soc_base", how="inner").merge(
        major, on="major_group_code", how="left"
    )

    return clean_occupation_data(merged)


def clean_occupation_data(df):
    cleaned = df.copy()

    cleaned["WageGroup"] = cleaned["WageGroup"].fillna("Unspecified")
    cleaned["SalaryUnit"] = "Annual"
    cleaned.loc[cleaned["MedianSalary"] < 100, "SalaryUnit"] = "Hourly"
    cleaned["MedianSalaryAnnualized"] = cleaned["MedianSalary"]
    cleaned.loc[cleaned["SalaryUnit"] == "Hourly", "MedianSalaryAnnualized"] = (
        cleaned.loc[cleaned["SalaryUnit"] == "Hourly", "MedianSalary"] * 2080
    )

    cleaned["ChanceAutoClean"] = cleaned["ChanceAuto"].replace(-1, np.nan)
    cleaned["JobZoneClean"] = cleaned["JobZone"].replace(-1, np.nan)
    cleaned["JobZoneLabel"] = cleaned["JobZoneClean"].map(JOB_ZONE_LABELS).fillna("Unknown")

    cleaned["ExposureGroup"] = np.where(
        cleaned["observed_exposure"] > 0, "Positive Exposure", "No Exposure"
    )
    cleaned["BrightLabel"] = np.where(cleaned["isBright"], "Bright Outlook", "Not Bright")
    cleaned["GreenLabel"] = np.where(cleaned["isGreen"], "Green", "Not Green")

    positive = cleaned.loc[cleaned["observed_exposure"] > 0, "observed_exposure"]
    if not positive.empty:
        q1 = positive.quantile(0.25)
        q3 = positive.quantile(0.75)
        cleaned["ExposureIntensity"] = "No Exposure"
        cleaned.loc[
            (cleaned["observed_exposure"] > 0) & (cleaned["observed_exposure"] <= q1),
            "ExposureIntensity",
        ] = "Low Positive Exposure"
        cleaned.loc[
            (cleaned["observed_exposure"] > q1) & (cleaned["observed_exposure"] < q3),
            "ExposureIntensity",
        ] = "Medium Positive Exposure"
        cleaned.loc[
            cleaned["observed_exposure"] >= q3,
            "ExposureIntensity",
        ] = "High Positive Exposure"
    else:
        cleaned["ExposureIntensity"] = "No Exposure"

    return cleaned


def get_numeric_summary(df):
    numeric_df = df.select_dtypes(include=["number"])
    summary = numeric_df.describe().T
    summary["median"] = numeric_df.median()
    try:
        summary["mode"] = numeric_df.mode().iloc[0]
    except Exception:
        summary["mode"] = np.nan
    ordered = ["count", "mean", "median", "mode", "std", "min", "25%", "50%", "75%", "max"]
    return summary[ordered].round(4)


def get_data_quality_summary(df):
    rows = []
    for column in df.columns:
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_values": int(df[column].isna().sum()),
                "missing_pct": round(float(df[column].isna().mean() * 100), 2),
                "unique_values": int(df[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def dataset_story(df):
    story = {}

    top_exposed = df.sort_values("observed_exposure", ascending=False).head(1)
    if not top_exposed.empty:
        story["top_occupation"] = top_exposed.iloc[0]["title"]
        story["top_occupation_exposure"] = float(top_exposed.iloc[0]["observed_exposure"])

    family_exposure = df.groupby("JobFamily")["observed_exposure"].mean().sort_values(
        ascending=False
    )
    if not family_exposure.empty:
        story["top_family"] = family_exposure.index[0]
        story["top_family_exposure"] = float(family_exposure.iloc[0])

    major_counts = df["major_group_title"].value_counts()
    if not major_counts.empty:
        story["largest_major_group"] = major_counts.index[0]
        story["largest_major_group_count"] = int(major_counts.iloc[0])

    return story


def safe_shapiro(values):
    if len(values) < 3:
        return None
    sample = values
    if len(values) > 500:
        sample = values[:500]
    return float(stats.shapiro(sample)[1])


def cohen_d(sample_a, sample_b):
    n1 = len(sample_a)
    n2 = len(sample_b)
    if n1 < 2 or n2 < 2:
        return None

    var1 = np.var(sample_a, ddof=1)
    var2 = np.var(sample_b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / float(n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(sample_a) - np.mean(sample_b)) / pooled)


def run_t_test(df):
    subset = df[["ExposureGroup", "MedianSalaryAnnualized"]].dropna()
    no_exp = subset.loc[
        subset["ExposureGroup"] == "No Exposure", "MedianSalaryAnnualized"
    ].astype(float).values
    pos_exp = subset.loc[
        subset["ExposureGroup"] == "Positive Exposure", "MedianSalaryAnnualized"
    ].astype(float).values

    results = {
        "mean_no": float(np.mean(no_exp)) if len(no_exp) else None,
        "mean_pos": float(np.mean(pos_exp)) if len(pos_exp) else None,
        "n_no": int(len(no_exp)),
        "n_pos": int(len(pos_exp)),
        "normality_no": None,
        "normality_pos": None,
        "variance_p": None,
        "t_stat": None,
        "p_value": None,
        "effect_size": None,
        "interpretation": "Not enough data to run the t-test.",
    }

    if len(no_exp) < 2 or len(pos_exp) < 2:
        return results

    results["normality_no"] = safe_shapiro(no_exp)
    results["normality_pos"] = safe_shapiro(pos_exp)
    results["variance_p"] = float(stats.levene(no_exp, pos_exp)[1])

    equal_var = results["variance_p"] is not None and results["variance_p"] >= 0.05
    t_stat, p_value = stats.ttest_ind(no_exp, pos_exp, equal_var=equal_var)
    results["t_stat"] = float(t_stat)
    results["p_value"] = float(p_value)
    results["effect_size"] = cohen_d(no_exp, pos_exp)

    if p_value < 0.05:
        results["interpretation"] = (
            "Occupations with positive observed AI exposure have a statistically different "
            "mean annualized salary than occupations with no observed AI exposure."
        )
    else:
        results["interpretation"] = (
            "There is not enough evidence to conclude that positive-exposure and no-exposure "
            "occupations differ in mean annualized salary."
        )

    return results


def run_group_t_test(df, value_column, question_label):
    subset = df[["ExposureGroup", value_column]].dropna()
    no_exp = subset.loc[subset["ExposureGroup"] == "No Exposure", value_column].astype(float).values
    pos_exp = subset.loc[
        subset["ExposureGroup"] == "Positive Exposure", value_column
    ].astype(float).values

    results = {
        "question": question_label,
        "value_column": value_column,
        "mean_no": float(np.mean(no_exp)) if len(no_exp) else None,
        "mean_pos": float(np.mean(pos_exp)) if len(pos_exp) else None,
        "n_no": int(len(no_exp)),
        "n_pos": int(len(pos_exp)),
        "normality_no": None,
        "normality_pos": None,
        "variance_p": None,
        "t_stat": None,
        "p_value": None,
        "effect_size": None,
        "interpretation": "Not enough data to run the t-test.",
    }

    if len(no_exp) < 2 or len(pos_exp) < 2:
        return results

    results["normality_no"] = safe_shapiro(no_exp)
    results["normality_pos"] = safe_shapiro(pos_exp)
    results["variance_p"] = float(stats.levene(no_exp, pos_exp)[1])

    equal_var = results["variance_p"] is not None and results["variance_p"] >= 0.05
    t_stat, p_value = stats.ttest_ind(no_exp, pos_exp, equal_var=equal_var)
    results["t_stat"] = float(t_stat)
    results["p_value"] = float(p_value)
    results["effect_size"] = cohen_d(no_exp, pos_exp)

    if p_value < 0.05:
        results["interpretation"] = (
            "Occupations with positive observed AI exposure have a statistically different "
            "mean value for {} than occupations with no observed AI exposure."
        ).format(question_label.lower())
    else:
        results["interpretation"] = (
            "There is not enough evidence to conclude that positive-exposure and no-exposure "
            "occupations differ in mean {}."
        ).format(question_label.lower())

    return results


def train_exposure_model(df):
    model_df = df.copy()
    model_df["has_positive_exposure"] = (model_df["observed_exposure"] > 0).astype(int)

    feature_columns = [
        "JobFamily",
        "major_group_title",
        "isBright",
        "isGreen",
        "JobZoneLabel",
        "MedianSalaryAnnualized",
        "JobForecast",
        "ChanceAutoClean",
        "WageGroup",
    ]
    X = model_df[feature_columns]
    y = model_df["has_positive_exposure"]

    numeric_features = ["MedianSalaryAnnualized", "JobForecast", "ChanceAutoClean"]
    categorical_features = ["JobFamily", "major_group_title", "JobZoneLabel", "WageGroup"]
    boolean_features = ["isBright", "isGreen"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            ("bool", "passthrough", boolean_features),
        ]
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(y_test, pred),
            index=["Actual No Exposure", "Actual Positive Exposure"],
            columns=["Predicted No Exposure", "Predicted Positive Exposure"],
        ),
    }

    feature_names = []
    feature_names.extend(numeric_features)
    onehot = pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    if hasattr(onehot, "get_feature_names_out"):
        cat_names = onehot.get_feature_names_out(categorical_features).tolist()
    else:
        cat_names = onehot.get_feature_names(categorical_features).tolist()
    feature_names.extend(cat_names)
    feature_names.extend(boolean_features)

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": pipeline.named_steps["model"].feature_importances_}
    ).sort_values("importance", ascending=False)

    return metrics, importance_df.head(12)


def train_salary_model(df):
    model_df = df.copy()

    feature_columns = [
        "observed_exposure",
        "JobFamily",
        "major_group_title",
        "isBright",
        "isGreen",
        "JobZoneLabel",
        "JobForecast",
        "ChanceAutoClean",
        "WageGroup",
    ]
    target_column = "MedianSalaryAnnualized"

    X = model_df[feature_columns]
    y = model_df[target_column]

    numeric_features = ["observed_exposure", "JobForecast", "ChanceAutoClean"]
    categorical_features = ["JobFamily", "major_group_title", "JobZoneLabel", "WageGroup"]
    boolean_features = ["isBright", "isGreen"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            ("bool", "passthrough", boolean_features),
        ]
    )

    model = LinearRegression()
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": rmse,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    feature_names = []
    feature_names.extend(numeric_features)
    onehot = pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    if hasattr(onehot, "get_feature_names_out"):
        cat_names = onehot.get_feature_names_out(categorical_features).tolist()
    else:
        cat_names = onehot.get_feature_names(categorical_features).tolist()
    feature_names.extend(cat_names)
    feature_names.extend(boolean_features)

    coefficients = pipeline.named_steps["model"].coef_
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    # Fit an OLS model on the transformed full dataset to get coefficient p-values.
    X_full_transformed = pipeline.named_steps["preprocess"].fit_transform(X)
    if hasattr(X_full_transformed, "toarray"):
        X_full_transformed = X_full_transformed.toarray()
    X_full_transformed = pd.DataFrame(X_full_transformed, columns=feature_names)
    X_full_transformed = sm.add_constant(X_full_transformed, has_constant="add")
    ols_model = sm.OLS(y, X_full_transformed).fit()

    pvalue_map = ols_model.pvalues.to_dict()
    coef_df["p_value"] = coef_df["feature"].map(pvalue_map)
    coef_df["significant_0_05"] = coef_df["p_value"] < 0.05

    exposure_coef = coef_df.loc[coef_df["feature"] == "observed_exposure", "coefficient"]
    exposure_coef_value = float(exposure_coef.iloc[0]) if not exposure_coef.empty else None
    exposure_pvalue = coef_df.loc[coef_df["feature"] == "observed_exposure", "p_value"]
    exposure_pvalue_value = float(exposure_pvalue.iloc[0]) if not exposure_pvalue.empty else None

    interpretation = None
    if exposure_coef_value is not None:
        interpretation = (
            "In this linear model, a 0.1 increase in observed exposure is associated with an "
            "estimated ${:,.0f} change in annualized salary, holding the other included features constant."
        ).format(exposure_coef_value * 0.1)
        if exposure_pvalue_value is not None:
            if exposure_pvalue_value < 0.05:
                interpretation += " The exposure coefficient is statistically significant at the 0.05 level."
            else:
                interpretation += " The exposure coefficient is not statistically significant at the 0.05 level."

    return (
        metrics,
        coef_df.drop(columns=["abs_coefficient"]).head(15),
        exposure_coef_value,
        exposure_pvalue_value,
        interpretation,
    )


def percentile_score(series, value):
    clean = series.dropna().astype(float)
    if clean.empty or value is None or pd.isna(value):
        return None
    return float((clean <= float(value)).mean() * 100)


def classify_career_profile(disruption_score, opportunity_score, exposure_score):
    if disruption_score is None or opportunity_score is None:
        return "Profile unavailable"
    if disruption_score >= 70 and opportunity_score < 45:
        return "Higher automation risk"
    if disruption_score >= 55 and opportunity_score >= 45:
        return "AI reshaping with mixed outlook"
    if exposure_score is not None and exposure_score >= 50:
        return "AI augmented opportunity"
    return "Lower near-term AI pressure"


def build_career_profile(df, occupation_title):
    row = df.loc[df["title"] == occupation_title].head(1)
    if row.empty:
        return None

    row = row.iloc[0]

    exposure_score = percentile_score(df["observed_exposure"], row["observed_exposure"])
    automation_score = percentile_score(df["ChanceAutoClean"], row["ChanceAutoClean"])
    salary_score = percentile_score(df["MedianSalaryAnnualized"], row["MedianSalaryAnnualized"])
    outlook_score = percentile_score(df["JobForecast"], row["JobForecast"])
    bright_score = 100.0 if bool(row["isBright"]) else 0.0

    disruption_score = None
    if exposure_score is not None and automation_score is not None:
        disruption_score = float(0.45 * exposure_score + 0.55 * automation_score)

    opportunity_score = None
    if salary_score is not None and outlook_score is not None:
        opportunity_score = float(0.4 * salary_score + 0.35 * outlook_score + 0.25 * bright_score)

    profile_label = classify_career_profile(disruption_score, opportunity_score, exposure_score)

    score_rows = [
        {"score_type": "AI exposure", "score": exposure_score},
        {"score_type": "Automation risk", "score": automation_score},
        {"score_type": "Salary strength", "score": salary_score},
        {"score_type": "Outlook strength", "score": outlook_score},
        {"score_type": "Career opportunity", "score": opportunity_score},
        {"score_type": "AI disruption", "score": disruption_score},
    ]
    score_df = pd.DataFrame(score_rows)

    narrative = []
    if exposure_score is not None:
        if exposure_score >= 75:
            narrative.append("This occupation sits in the high end of AI exposure in the dataset.")
        elif exposure_score >= 40:
            narrative.append("This occupation shows a moderate level of AI exposure.")
        else:
            narrative.append("This occupation shows relatively low observed AI exposure.")

    if automation_score is not None:
        if automation_score >= 75:
            narrative.append("Its automation chance is high relative to other occupations.")
        elif automation_score >= 40:
            narrative.append("Its automation chance is moderate relative to other occupations.")
        else:
            narrative.append("Its automation chance is on the lower side of the dataset.")

    if opportunity_score is not None:
        if opportunity_score >= 70:
            narrative.append("Salary and outlook indicators are comparatively strong.")
        elif opportunity_score >= 45:
            narrative.append("Salary and outlook indicators are mixed but not weak.")
        else:
            narrative.append("Salary and outlook indicators are relatively weak compared with other occupations.")

    return {
        "title": row["title"],
        "job_family": row["JobFamily"],
        "major_group_title": row["major_group_title"],
        "salary": float(row["MedianSalaryAnnualized"]) if pd.notna(row["MedianSalaryAnnualized"]) else None,
        "forecast": float(row["JobForecast"]) if pd.notna(row["JobForecast"]) else None,
        "observed_exposure": float(row["observed_exposure"]) if pd.notna(row["observed_exposure"]) else None,
        "automation_chance": float(row["ChanceAutoClean"]) if pd.notna(row["ChanceAutoClean"]) else None,
        "bright_label": row["BrightLabel"],
        "green_label": row["GreenLabel"],
        "job_zone_label": row["JobZoneLabel"],
        "profile_label": profile_label,
        "score_df": score_df,
        "narrative": narrative,
    }
