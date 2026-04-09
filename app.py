import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.occupation_analysis import build_occupation_dataset
from src.occupation_analysis import dataset_story
from src.occupation_analysis import get_data_quality_summary
from src.occupation_analysis import get_numeric_summary
from src.occupation_analysis import run_t_test
from src.occupation_analysis import train_exposure_model
from src.occupation_analysis import train_salary_model


PLOT_TEMPLATE = "plotly_white"


st.set_page_config(page_title="Occupation AI Exposure Dashboard", layout="wide")


def metric_value(value, decimals=3):
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        fmt = "{:,.%df}" % decimals
        return fmt.format(value)
    return "{:,}".format(value)


@st.cache_data
def load_dataset(base_dir):
    return build_occupation_dataset(base_dir)


@st.cache_data
def get_model_outputs(df):
    return train_exposure_model(df)


@st.cache_data
def get_salary_model_outputs(df):
    return train_salary_model(df)


st.title("Occupation-Level AI Exposure Dashboard")
st.caption(
    "A clean occupation-focused analysis of observed AI exposure, salary, outlook, job preparation, and automation risk."
)

with st.expander("What the main terms mean", expanded=False):
    st.markdown("- `observed_exposure`: Anthropic's occupation-level AI exposure signal")
    st.markdown("- `Job Zone`: O*NET preparation level based on education, experience, and training")
    st.markdown("- `Bright Outlook`: occupations expected to grow quickly or have many openings")
    st.markdown("- `Green`: occupations connected to green-economy activity")
    st.markdown("- `ChanceAuto`: automation-chance value from the wage dataset, with `-1` treated as unknown")
    st.markdown("- `Positive Exposure`: occupations with `observed_exposure > 0`")
    st.markdown("- `No Exposure`: occupations with `observed_exposure = 0`")

default_data_dir = "data"
st.sidebar.header("Data")
data_dir = st.sidebar.text_input("Data directory", value=default_data_dir)

required_files = ["job_exposure.csv", "wage_data.csv", "SOC_Structure.csv"]
missing = [name for name in required_files if not os.path.exists(os.path.join(data_dir, name))]
if missing:
    st.warning("Missing required files: {}".format(", ".join(missing)))
    st.stop()

df = load_dataset(data_dir)

st.sidebar.header("Filters")
families = sorted(df["JobFamily"].dropna().unique().tolist())
selected_families = st.sidebar.multiselect("Job families", families, default=families)

exposure_choices = sorted(df["ExposureGroup"].dropna().unique().tolist())
selected_exposure = st.sidebar.multiselect("Exposure group", exposure_choices, default=exposure_choices)

zone_choices = ["All"] + sorted(df["JobZoneLabel"].dropna().unique().tolist())
selected_zone = st.sidebar.selectbox("Job zone", zone_choices, index=0)

filtered_df = df[df["JobFamily"].isin(selected_families) & df["ExposureGroup"].isin(selected_exposure)].copy()
if selected_zone != "All":
    filtered_df = filtered_df[filtered_df["JobZoneLabel"] == selected_zone]

if filtered_df.empty:
    st.error("No rows match the current filters. Broaden the filters and try again.")
    st.stop()

story = dataset_story(filtered_df)

st.subheader("Overview")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Occupations", metric_value(len(filtered_df), 0))
m2.metric("Avg Exposure", metric_value(filtered_df["observed_exposure"].mean(), 4))
m3.metric("Avg Annualized Salary", "${}".format(metric_value(filtered_df["MedianSalaryAnnualized"].mean(), 0)))
m4.metric("Positive Exposure Share", "{}%".format(metric_value((filtered_df["observed_exposure"] > 0).mean() * 100, 1)))
m5.metric("Avg Automation Chance", "{}%".format(metric_value(filtered_df["ChanceAutoClean"].mean(), 1)))

i1, i2, i3 = st.columns(3)
i1.info(
    "Most exposed occupation: {} ({})".format(
        story.get("top_occupation", "N/A"),
        metric_value(story.get("top_occupation_exposure"), 4),
    )
)
i2.info(
    "Highest-exposure family: {} ({})".format(
        story.get("top_family", "N/A"),
        metric_value(story.get("top_family_exposure"), 4),
    )
)
i3.info(
    "Largest major group in view: {} ({} occupations)".format(
        story.get("largest_major_group", "N/A"),
        metric_value(story.get("largest_major_group_count"), 0),
    )
)

with st.expander("Preview merged dataset", expanded=False):
    st.dataframe(filtered_df.head(25), use_container_width=True)

tabs = st.tabs(
    [
        "Summary",
        "EDA",
        "Hypothesis Test",
        "ML: Exposure Classification",
        "ML: Salary Regression",
    ]
)

with tabs[0]:
    st.markdown("**Data validation and summary**")
    validation_left, validation_right = st.columns(2)
    with validation_left:
        st.dataframe(get_data_quality_summary(filtered_df), use_container_width=True, height=320)
    with validation_right:
        st.dataframe(get_numeric_summary(filtered_df), use_container_width=True, height=320)

    summary_left, summary_right = st.columns(2)
    with summary_left:
        exposure_counts = filtered_df["ExposureGroup"].value_counts().reset_index()
        exposure_counts.columns = ["ExposureGroup", "Count"]
        summary_fig = px.bar(
            exposure_counts,
            x="ExposureGroup",
            y="Count",
            color="ExposureGroup",
            title="Exposure group counts",
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(summary_fig, use_container_width=True)

    with summary_right:
        family_preview = (
            filtered_df.groupby("JobFamily")["observed_exposure"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .head(10)
        )
        family_preview_fig = px.bar(
            family_preview.sort_values("observed_exposure"),
            x="observed_exposure",
            y="JobFamily",
            orientation="h",
            color="observed_exposure",
            title="Top 10 job families by average exposure",
            template=PLOT_TEMPLATE,
            color_continuous_scale="Tealgrn",
        )
        st.plotly_chart(family_preview_fig, use_container_width=True)

with tabs[1]:
    st.markdown("**Exploratory Data Analysis**")
    top_left, top_right = st.columns(2)
    with top_left:
        family_exposure = (
            filtered_df.groupby("JobFamily")["observed_exposure"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        family_fig = px.bar(
            family_exposure,
            x="observed_exposure",
            y="JobFamily",
            orientation="h",
            color="observed_exposure",
            title="Average observed exposure by job family",
            template=PLOT_TEMPLATE,
            color_continuous_scale="Tealgrn",
        )
        family_fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(family_fig, use_container_width=True)

    with top_right:
        major_exposure = (
            filtered_df.groupby("major_group_title")["observed_exposure"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .head(15)
        )
        major_fig = px.bar(
            major_exposure.sort_values("observed_exposure"),
            x="observed_exposure",
            y="major_group_title",
            orientation="h",
            color="observed_exposure",
            title="Top major occupation groups by exposure",
            template=PLOT_TEMPLATE,
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(major_fig, use_container_width=True)

    mid_left, mid_right = st.columns(2)
    with mid_left:
        scatter_fig = px.scatter(
            filtered_df,
            x="MedianSalaryAnnualized",
            y="observed_exposure",
            color="ExposureGroup",
            size="JobForecast",
            hover_name="title",
            hover_data=["JobFamily", "JobZoneLabel", "BrightLabel", "GreenLabel"],
            title="Salary vs observed AI exposure",
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    with mid_right:
        auto_scatter = px.scatter(
            filtered_df.dropna(subset=["ChanceAutoClean"]),
            x="ChanceAutoClean",
            y="observed_exposure",
            color="JobFamily",
            hover_name="title",
            title="Automation chance vs observed AI exposure",
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(auto_scatter, use_container_width=True)

    low_left, low_right = st.columns(2)
    with low_left:
        exposure_box = px.box(
            filtered_df,
            x="ExposureGroup",
            y="MedianSalaryAnnualized",
            color="ExposureGroup",
            points="outliers",
            title="Salary by exposure group",
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(exposure_box, use_container_width=True)

    with low_right:
        zone_heat = pd.pivot_table(
            filtered_df,
            index="JobFamily",
            columns="JobZoneLabel",
            values="observed_exposure",
            aggfunc="mean",
        )
        heat_fig = px.imshow(
            zone_heat,
            color_continuous_scale="YlGnBu",
            aspect="auto",
            title="Average exposure by job family and job zone",
        )
        st.plotly_chart(heat_fig, use_container_width=True)

    rank_left, rank_right = st.columns(2)
    with rank_left:
        top_exposed = filtered_df.sort_values("observed_exposure", ascending=False).head(10)[
            ["title", "JobFamily", "observed_exposure", "MedianSalaryAnnualized", "BrightLabel"]
        ]
        st.markdown("**Top 10 most exposed occupations**")
        st.dataframe(top_exposed, use_container_width=True)

    with rank_right:
        zero_exposed = filtered_df.sort_values(["observed_exposure", "MedianSalaryAnnualized"], ascending=[True, False]).head(10)[
            ["title", "JobFamily", "observed_exposure", "MedianSalaryAnnualized", "BrightLabel"]
        ]
        st.markdown("**Example low- or zero-exposure occupations**")
        st.dataframe(zero_exposed, use_container_width=True)

with tabs[2]:
    st.markdown(
        "**Research question:** Do occupations with positive observed AI exposure have a different mean annualized salary than occupations with no observed AI exposure?"
    )
    st.markdown(
        "`H0`: The mean annualized salary is the same for no-exposure and positive-exposure occupations.  "
        "`Ha`: The mean annualized salary is different for the two groups."
    )

    test_results = run_t_test(filtered_df)

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Mean No Exposure", "${}".format(metric_value(test_results.get("mean_no"), 0)))
    h2.metric("Mean Positive Exposure", "${}".format(metric_value(test_results.get("mean_pos"), 0)))
    h3.metric("t-statistic", metric_value(test_results.get("t_stat"), 4))
    h4.metric("p-value", metric_value(test_results.get("p_value"), 4))

    s1, s2, s3 = st.columns(3)
    s1.metric("No Exposure n", metric_value(test_results.get("n_no"), 0))
    s2.metric("Positive Exposure n", metric_value(test_results.get("n_pos"), 0))
    s3.metric("Effect Size", metric_value(test_results.get("effect_size"), 4))

    assumptions = pd.DataFrame(
        [
            {"check": "Shapiro p-value (No Exposure)", "value": test_results.get("normality_no")},
            {"check": "Shapiro p-value (Positive Exposure)", "value": test_results.get("normality_pos")},
            {"check": "Levene p-value", "value": test_results.get("variance_p")},
        ]
    )
    st.dataframe(assumptions, use_container_width=True)

    decision = (
        "Reject H0"
        if test_results.get("p_value") is not None and test_results["p_value"] < 0.05
        else "Fail to reject H0"
    )
    st.markdown("**Decision:** {}".format(decision))
    st.write(test_results.get("interpretation"))

    compare_fig = px.violin(
        filtered_df,
        x="ExposureGroup",
        y="MedianSalaryAnnualized",
        color="ExposureGroup",
        box=True,
        points="all",
        title="Annualized salary by exposure group",
        template=PLOT_TEMPLATE,
    )
    st.plotly_chart(compare_fig, use_container_width=True)

with tabs[3]:
    st.markdown("**ML model 1: classify whether an occupation has positive observed AI exposure**")
    metrics, importance_df = get_model_outputs(df)

    ml1, ml2, ml3, ml4 = st.columns(4)
    ml1.metric("Accuracy", metric_value(metrics.get("accuracy"), 3))
    ml2.metric("Precision", metric_value(metrics.get("precision"), 3))
    ml3.metric("Recall", metric_value(metrics.get("recall"), 3))
    ml4.metric("F1 Score", metric_value(metrics.get("f1"), 3))

    split1, split2 = st.columns(2)
    split1.metric("Training Rows", metric_value(metrics.get("train_rows"), 0))
    split2.metric("Test Rows", metric_value(metrics.get("test_rows"), 0))

    class_left, class_right = st.columns(2)
    with class_left:
        st.markdown("**Confusion Matrix**")
        st.dataframe(metrics.get("confusion_matrix"), use_container_width=True)

    with class_right:
        importance_fig = px.bar(
            importance_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            title="Top exposure-classification features",
            template=PLOT_TEMPLATE,
            color_continuous_scale="OrRd",
        )
        st.plotly_chart(importance_fig, use_container_width=True)

with tabs[4]:
    st.markdown("**ML model 2: predict annualized salary with linear regression**")
    (
        salary_metrics,
        coef_df,
        exposure_coef,
        exposure_pvalue,
        salary_interpretation,
    ) = get_salary_model_outputs(df)

    reg1, reg2, reg3 = st.columns(3)
    reg1.metric("R²", metric_value(salary_metrics.get("r2"), 3))
    reg2.metric("MAE", "${}".format(metric_value(salary_metrics.get("mae"), 0)))
    reg3.metric("RMSE", "${}".format(metric_value(salary_metrics.get("rmse"), 0)))

    reg_left, reg_right = st.columns(2)
    with reg_left:
        st.markdown("**Largest salary-model coefficients with p-values**")
        st.dataframe(coef_df, use_container_width=True)

    with reg_right:
        coef_chart_df = coef_df.head(12).copy().sort_values("coefficient")
        coef_fig = px.bar(
            coef_chart_df,
            x="coefficient",
            y="feature",
            orientation="h",
            color="coefficient",
            title="Top salary-model coefficients",
            template=PLOT_TEMPLATE,
            color_continuous_scale="RdBu",
        )
        st.plotly_chart(coef_fig, use_container_width=True)

    exposure_text = (
        "${:,.0f}".format(exposure_coef) if exposure_coef is not None else "N/A"
    )
    exposure_p_text = metric_value(exposure_pvalue, 4) if exposure_pvalue is not None else "N/A"
    st.markdown(
        "**Observed exposure coefficient:** {}".format(exposure_text)
    )
    st.markdown(
        "**Observed exposure p-value:** {}".format(exposure_p_text)
    )
    if salary_interpretation:
        st.info(salary_interpretation)
