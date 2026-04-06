#!/usr/bin/env bash
set -euo pipefail

mkdir -p data

curl -L -s https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/labor_market_impacts/job_exposure.csv -o data/job_exposure.csv
curl -L -s https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/release_2025_02_10/wage_data.csv -o data/wage_data.csv
curl -L -s https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/release_2025_02_10/SOC_Structure.csv -o data/SOC_Structure.csv

echo "Downloaded:"
echo "  data/job_exposure.csv"
echo "  data/wage_data.csv"
echo "  data/SOC_Structure.csv"
