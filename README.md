# Occupation-Level AI Exposure Analysis

Interactive Streamlit project for analyzing how AI exposure varies across occupations using Anthropic's Economic Index occupation exposure data plus occupation metadata from the same repository.

## Project Goal

The project asks a simple question:

**Which occupations appear more exposed to AI, and how do salary, outlook, job preparation, and automation risk differ across those occupations?**

The app supports:

- data validation and summary statistics
- exploratory data analysis
- hypothesis testing
- machine learning
- an interactive presentation layer in Streamlit
- a single-occupation career insight view
- a definitions tab for non-technical readers

## Dataset Design

This project builds a clean **one-row-per-occupation** dataset from three source files:

- `job_exposure.csv`
- `wage_data.csv`
- `SOC_Structure.csv`

The merge strategy is intentionally conservative:

1. `job_exposure.csv` provides Anthropic's occupation-level `observed_exposure` score.
2. `wage_data.csv` is filtered to canonical occupation rows where `SOCcode` ends in `.00`.
3. `SOC_Structure.csv` adds major occupation-group labels.

This produces a merged dataset with:

- **672 rows**
- **16 raw columns**
- **25 columns after derived features are added**

## Main Variables

- `observed_exposure`: Anthropic's occupation-level AI exposure signal
- `MedianSalaryAnnualized`: annualized salary used in analysis
- `ChanceAutoClean`: automation score with `-1` treated as unknown
- `JobZoneLabel`: readable job preparation category
- `ExposureGroup`: `No Exposure` vs `Positive Exposure`
- `ExposureIntensity`: no, low, medium, or high positive exposure
- `major_group_title`: major occupation-group label from SOC structure

## What the App Includes

### EDA

- data quality table and summary statistics
- average AI exposure by job family
- top major occupation groups by exposure
- salary vs observed exposure
- automation chance vs observed exposure
- salary comparison across exposure groups
- exposure heatmap by family and job zone
- top exposed occupations and low-exposure occupations

### Hypothesis Test

Research question 1:

**Do occupations with positive observed AI exposure have a different mean annualized salary than occupations with no observed AI exposure?**

- `H0`: the mean annualized salary is the same in both groups
- `Ha`: the mean annualized salary is different

Research question 2:

**Do occupations with positive observed AI exposure have a different mean automation chance than occupations with no observed AI exposure?**

- `H0`: the mean automation chance is the same in both groups
- `Ha`: the mean automation chance is different

### Machine Learning

Model 1: classification

- target: whether an occupation has positive AI exposure
- model: `RandomForestClassifier`
- features: job family, major group, bright outlook, green-job flag, job zone, salary, job forecast, automation chance, and wage group

Model 2: regression

- target: `MedianSalaryAnnualized`
- model: `LinearRegression`
- includes `observed_exposure` as a direct predictor
- used to interpret how AI exposure is associated with salary after accounting for the other included variables
- includes coefficient p-values from an OLS fit for significance-based interpretation

### Dashboard tabs

The app is organized into seven tabs:

- `Summary`
- `EDA`
- `Hypothesis Test`
- `ML: Exposure Classification`
- `ML: Salary Regression`
- `Career Insight`
- `Definitions`

The current UI uses a custom visual layer on top of Streamlit with a warm light theme, styled metric cards, unified Plotly charts, and guided section notes so the dashboard is easier to present live.

## Repository Structure

```text
final/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── .gitkeep
├── scripts/
│   └── download_data.sh
└── src/
    ├── __init__.py
    └── occupation_analysis.py
```

## Setup

### 1. Create the environment

```bash
conda create -n final_project python=3.11 -y
conda activate final_project
pip install -r requirements.txt
```

### 2. Download the source data

The CSV files are not meant to be committed to git. Download them into `data/` with:

```bash
bash scripts/download_data.sh
```

This downloads:

- `data/job_exposure.csv`
- `data/wage_data.csv`
- `data/SOC_Structure.csv`

### 3. Run the app

```bash
streamlit run app.py
```

### Optional OpenAI powered automation assessment

The Career Insight tab includes an optional OpenAI call that generates an LLM adjusted automation score, a short explanation, and nearby transition advice for the selected occupation.

Set your API key in the shell before starting Streamlit:

```bash
export OPENAI_API_KEY="your_api_key_here"
streamlit run app.py
```

The app reads the key from the `OPENAI_API_KEY` environment variable. The raw dataset automation score remains visible, and the LLM assessment is generated only when you click the button in the Career Insight tab.

## Data Sources

- Anthropic Economic Index repository:
  <https://huggingface.co/datasets/Anthropic/EconomicIndex>
- Job exposure file:
  <https://huggingface.co/datasets/Anthropic/EconomicIndex/blob/main/labor_market_impacts/job_exposure.csv>
- Wage data file:
  <https://huggingface.co/datasets/Anthropic/EconomicIndex/blob/main/release_2025_02_10/wage_data.csv>
- SOC structure file:
  <https://huggingface.co/datasets/Anthropic/EconomicIndex/blob/main/release_2025_02_10/SOC_Structure.csv>

## Notes

- `observed_exposure` is the core AI-related variable in the project.
- Positive exposure does **not** automatically mean job replacement; it indicates stronger association with AI-related task activity in Anthropic's data.
- The app rebuilds the merged dataset at runtime from the three source CSV files.
