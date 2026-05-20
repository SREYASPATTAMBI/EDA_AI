# 🧠 EDA Pilot
> **Drop any CSV file. Get a trained AI model. No data science knowledge required.**

EDA Pilot is an AI-powered data analysis tool that talks to you like a human. It looks at your data, explains what it finds in plain English, fixes problems automatically, and builds a machine learning model — all through a simple conversation in your terminal.

---

## 🤔 What Problem Does This Solve?

Data science has a massive barrier to entry. Before you can even build a model, you need to:
- Clean messy data
- Handle missing values the right way
- Remove outliers
- Choose the right algorithm
- Validate your results

Most people give up here. EDA Pilot does all of this for you — automatically — while explaining every step so you actually understand what's happening.

---

## 🎬 How It Works

Just run the script. EDA Pilot will guide you through everything:

```
You: python analyst.py

EDA Pilot: ✅ Loaded your data — 500 rows, 12 columns
           There might be missing values. Should I check? (yes/no)

You: yes

EDA Pilot: Found 3 columns with missing data.
           Analyzing WHY the data is missing...
           → salary: Missing because of extreme values (MNAR)
           → age: Missing randomly (MCAR)
           Fixing them now using the right method for each...
```

It keeps going — standardizing your data, detecting outliers, building models, comparing them, and saving the best one — all while explaining what it's doing and why.

---

## ✨ What Makes It Special

### 🔬 Smart Missing Value Detection
Most tools just fill missing values with the average. That's lazy and often wrong.

EDA Pilot detects **why** your data is missing and picks the right fix:

| Pattern | What It Means | How EDA Pilot Fixes It |
|---------|--------------|----------------------|
| **MCAR** | Missing randomly — no pattern | Fills with mean or median |
| **MAR** | Missing because of another column | Uses KNN to fill intelligently |
| **MNAR** | Missing because of extreme values | Flags it and preserves the signal |

This is what professional data scientists do manually. EDA Pilot does it automatically.

### 🤖 5 Models Tested, Best One Recommended
EDA Pilot doesn't just pick one model and hope for the best. It tests 5 different algorithms using Cross Validation and recommends the most accurate one — but lets you override the choice.

### 🧠 AI That Explains Everything
Every result is explained in plain English by Gemma 2B (running locally on your machine). No jargon, no confusion — just clear explanations of what the data shows and what to do next.

---

## ⚙️ The 10 Phases

| Phase | What Happens |
|-------|-------------|
| **1** | Loads your CSV and gives a full overview of the data |
| **2** | Detects which columns have missing values |
| **3** | Analyzes WHY data is missing (MCAR / MAR / MNAR) and fixes it |
| **4** | Standardizes inconsistent text and number formats |
| **5** | Detects outliers in every column |
| **6** | Handles outliers intelligently — ignores ID columns automatically |
| **7** | Lets you choose which column you want to predict |
| **8** | Creates new useful columns from existing ones (feature engineering) |
| **9** | Tests 5 ML models with Cross Validation and recommends the best |
| **10** | Removes weak columns, trains the final model, saves it to disk |

---

## 📦 How To Run It

**Step 1 — Install Ollama and pull Gemma:**
```bash
ollama pull gemma2:2b
```

**Step 2 — Install Python dependencies:**
```bash
pip install pandas scikit-learn ollama
```

**Step 3 — Add your data:**
Put any CSV file in the folder and rename it `data.csv`

**Step 4 — Run:**
```bash
python analyst.py
```

That's it. EDA Pilot takes it from there.

---

## 💡 Works On Any CSV

Not built for one specific dataset. Drop in any CSV — HR data, sales data, student records, financial data — and EDA Pilot adapts automatically.

---

## 📁 Output Files

After running, you get:

| File | What It Is |
|------|-----------|
| `cleaned.csv` | Your data after missing values are fixed |
| `standardized_data.csv` | Your data after text and number cleanup |
| `final_cleaned_data.csv` | Your data after outlier removal |
| `model.pkl` | Your trained ML model ready to use |
| `selected_features.pkl` | The columns the model uses to predict |

---

## 🛠️ Built With

- **Python** — core language
- **Pandas** — data manipulation
- **Scikit-learn** — machine learning models and imputation
- **Ollama + Gemma 2B** — local LLM for plain English explanations

---

## 👤 Author

**Sreyas Pattambi** — Independent Researcher
[GitHub](https://github.com/SREYASPATTAMBI)