AI Data Analyst Project
# EDA Pilot (EDA_AI)

An automated Exploratory Data Analysis (EDA) and data-cleaning engine powered by Local LLMs. This tool analyzes tabular datasets, detects statistical anomalies, and leverages `gemma2:2b` via Ollama to generate python code dynamically for handling missing data and capping outliers.

## 🛠️ Features

- **Automated Ingestion & Profiling:** Loads CSV datasets and maps structures, shapes, missing values, and metadata.
- **Statistical Outlier Detection:** Automatically calculates Interquartile Range (IQR) bounds ($Q1 - 1.5 \times IQR$ to $Q3 + 1.5 \times IQR$) across numerical columns to identify anomaly counts.
- **AI-Driven Data Profiling:** Utilizes local LLMs to categorize statistical missingness types (MCAR, MAR, MNAR) and generate localized analyst reports.
- **Dynamic Missing Value Resolution:** Directs the LLM to write vector-based, non-looping Pandas cleanup code to sanitize mixed garbage datatypes (e.g., stripping non-numeric symbols like `Rs.`, `/5`) and perform median/mode imputation.
- **Outlier Capping (Winsorization):** Automatically generates code using the `.clip()` method to cap extreme values safely without losing data rows.

---

## 📦 Prerequisites & Tech Stack

- **Python 3.x**
- **Pandas**
- **Ollama** (with the `gemma2:2b` model pulled locally)

To pull the model before executing the script, run:
```bash
ollama pull gemma2:2b
