import pandas as pd
import ollama

# --- LOAD DATA ---
try:
    df = pd.read_csv("data.csv")
    print("✅ Successfully loaded data.csv")
except FileNotFoundError:
    print("❌ Error: data.csv not found. Please add it to the folder.")
    exit()
    
facts = {
    "column_names": df.columns.tolist(),
    "shape": f"{df.shape[0]} rows, {df.shape[1]} columns", 
    "info": df.dtypes.to_string(),
    "sample_data": df.head(2).to_string() 
}
prompt = f"""
You are a data scientist now you need to analyse the data :
{facts}
and give output :
sample_data : (FIRST 5 ROWS)
SHAPE:give me the exact number of columns and rows (eg.,500 rows,10 column)
INFO:(info)
COLUMN NAMES:(column_names)"""
print("🤖 Gemma is analysing the data")

response = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': prompt}])
print(response['message']['content'])

#--- asking the question and getting the input---
user_reply = input("\n There can be missing values.Should I check it (yes/no): ").lower().strip()
if user_reply=='yes':
    
    missing_values=df.isnull().sum().to_dict()
    
    prompt_missing_values= f"""
    The user said yes. Here is the technical count of missing values:
    {missing_values}
    
    just give the name of the column and missing values opposite to it.
    """
    print("\n🤖 gemma is checking for missing values now...")
    response_2 = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': prompt_missing_values}])
    print("\n" + response_2['message']['content'])
else:
    print("\n👍 Skipping the missing values check. Moving forward!")

# --- PHASE 3: DYNAMIC PATTERN ANALYSIS ---
user_reply_mcar = input("\nShould I analyze the missing data pattern (MCAR, MAR, MNAR)? (yes/no): ").lower().strip()

if user_reply_mcar == 'yes':
    missing_cols = [col for col in df.columns if df[col].isnull().any()]
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    if missing_cols and num_cols:
        pattern_report = {}

        for m_col in missing_cols:
            df['is_missing'] = df[m_col].isnull().astype(int)
            
            corrs = df[num_cols].corrwith(df['is_missing']).abs()
            if m_col in corrs: corrs = corrs.drop(m_col)

            if corrs.dropna().empty:
                pattern_report[m_col] = {
                    "tested_against": "none",
                    "mean_diff": "0.00%",
                    "verdict": "MCAR (Missing Completely at Random)"
                }
                continue

            rel_col = corrs.idxmax()
            
            stats = df.groupby('is_missing')[rel_col].mean()
            diff_pct = (abs(stats[0] - stats[1]) / stats[0]) * 100
            
            threshold_90 = df[rel_col].quantile(0.90)
            missing_in_top_tier = df[(df['is_missing'] == 1) & (df[rel_col] >= threshold_90)]
            
            if diff_pct < 5:
                verdict = "MCAR (Missing Completely at Random)"
            elif len(missing_in_top_tier) / (df['is_missing'].sum() + 1e-9) > 0.4:
                verdict = "MNAR (Missing Not At Random - Missingness is tied to extreme values)"
            else:
                verdict = f"MAR (Missing at Random - Pattern linked to {rel_col})"
            
            pattern_report[m_col] = {
                "tested_against": rel_col,
                "mean_diff": f"{diff_pct:.2f}%",
                "verdict": verdict
            }

        prompt_pattern = f"""
        Analyze these missing data patterns for the user: {pattern_report}
        
        Use the 5% threshold for MCAR vs MAR. 
        If it is MNAR, explain that the missing data is biased toward extreme high/low values.
        """
        
        print("\n🤖 Gemma is running pattern logic...")
        response_3 = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': prompt_pattern}])
        print("\n" + response_3['message']['content'])
        
        from sklearn.impute import KNNImputer, SimpleImputer

def fix_missing_data(df, pattern_report):
    print("\n🛠️ Starting Logic-Based Data Cleaning...")
    
    df_clean = df.copy()
    
    for col, result in pattern_report.items():
        verdict = result['verdict']
        
        if "MCAR" in verdict:
            if df_clean[col].dtype in ['int64', 'float64']:
                strategy = 'median' if abs(df_clean[col].skew()) > 1.5 else 'mean'
                imputer = SimpleImputer(strategy=strategy)
                df_clean[[col]] = imputer.fit_transform(df_clean[[col]])
                print(f"✅ {col} (MCAR): Filled with {strategy}")
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                print(f"✅ {col} (MCAR): Filled with Mode")

        elif "MAR" in verdict:
            if df_clean[col].dtype in ['float64', 'int64']:
                print(f"🔄 {col} (MAR): Running KNN Imputation...")
                imputer = KNNImputer(n_neighbors=5)
                num_cols = df_clean.select_dtypes(include=['number']).columns
                df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
                print(f"✅ {col} (MAR): Filled using KNN")
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                print(f"✅ {col} (MAR): Filled with Mode (text column)")

        elif "MNAR" in verdict:
            df_clean[f"{col}_was_missing"] = df_clean[col].isnull().astype(int)
            df_clean[col] = df_clean[col].fillna(-999) 
            print(f"✅ {col} (MNAR): Flagged and filled with -999")

    return df_clean

if user_reply_mcar == 'yes':
    df = fix_missing_data(df, pattern_report)
    df.to_csv("cleaned.csv", index=False)
    print("\n💾 Success! Cleaned data saved to 'cleaned.csv'")
# --- PHASE 4: SMART STANDARDIZATION ---
print("\n🤖 Gemma is analyzing the data structure for inconsistencies...")

data_sample = df.head(3).to_string()
standard_prompt = f"""
The user has zero data knowledge. Look at this data sample:
{data_sample}

Explain that some columns might have inconsistent text (like 'Pizza' vs 'pizza') 
or numbers trapped with symbols (like '$1,000' or '4.5/5'). 
Ask the user a simple 'yes' or 'no' question: should I 'Standardize' the data 
to make future analysis and math 100% accurate?
"""

response_4 = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': standard_prompt}])
print("\n" + response_4['message']['content'])

user_reply_std = input("\n(yes/no): ").lower().strip()

if user_reply_std == 'yes':
    print("\n🧼 Gemma is now standardizing the dataset...")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().str.lower()
            clean_test = df[col].str.replace(r'[^\d.]', '', regex=True)
            numeric_check = pd.to_numeric(clean_test, errors='coerce')
            if numeric_check.notnull().mean() > 0.6:
                df[col] = numeric_check
                print(f"💎 Column '{col}' converted from text to numbers.")
            else:
                print(f"📄 Column '{col}' cleaned as text.")
                
    print("\n✅ Standardization complete! Your data is now uniform.")
    df.to_csv("standardized_data.csv", index=False)
    print("💾 Progress saved to 'standardized_data.csv'")

else:
    print("\n👍 Understood. Proceeding with data in its original format.")
# --- PHASE 5: OUTLIER DETECTION ---
user_reply_out = input("\nShould I check for outliers in each column? (yes/no): ").lower().strip()

if user_reply_out == 'yes':
    print("\n🔍 Analyzing columns for extreme values and rare categories...")
    print("-" * 45)

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        print(f"{col} - {outlier_count} outliers (Numerical)")

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        counts = df[col].value_counts(normalize=True)
        rare_labels_count = counts[counts < 0.01].count()
        print(f"{col} - {rare_labels_count} rare categories (Categorical)")

    print("-" * 45)
    print("✅ Outlier analysis complete.")

else:
    print("\n👍 Skipping outlier detection.")
# --- PHASE 6: AI-DRIVEN OUTLIER HANDLING ---

all_cols = df.columns.tolist()
sample_vals = df.head(1).to_dict()

filter_prompt = f"""
You are a Data Architect. Look at these columns: {all_cols}
And this sample data: {sample_vals}

Identify columns that are:
1. Unique Identifiers (IDs, Codes)
2. Contact Info (Phone numbers, Emails)
3. Web Links (URLs)
4. Personal Names or Addresses

List ONLY the names of the columns that should be EXCLUDED from mathematical outlier detection.
Format: ["col1", "col2"]
"""

print("\n🤖 Gemma is determining which columns are 'Identity-based'...")
filter_res = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': filter_prompt}])
excluded_text = filter_res['message']['content']

import re
import json
match = re.search(r'\[.*\]', excluded_text)
excluded_cols = json.loads(match.group()) if match else []

num_cols = df.select_dtypes(include=['number']).columns.tolist()
final_outlier_targets = [c for c in num_cols if c not in excluded_cols]

question_prompt = f"""
I have analyzed the dataset. I will ignore columns like {excluded_cols} 
as they contain identity information. 

I found outliers in: {final_outlier_targets}. 
Ask the user in a friendly way if I should clear these outliers to improve 
the data quality for math and machine learning.
"""

response_out = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': question_prompt}])
print("\n" + response_out['message']['content'])

user_reply_fix_out = input("\n(yes/no): ").lower().strip()

if user_reply_fix_out == 'yes':
    print(f"\n🛠️ Standardizing {len(final_outlier_targets)} columns...")
    
    for col in final_outlier_targets:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"✅ {col}: Outliers capped.")

    df.to_csv("final_cleaned_data.csv", index=False)
    print("\n💾 Success! Dataset is now clean and saved to 'final_cleaned_data.csv'")
else:
    print("\n👍 Understood. Keeping all values as they are.")
# --- PHASE 7: TARGET COLUMN SELECTION ---

latest_file = "data.csv"
if user_reply_mcar == 'yes':
    latest_file = "cleaned.csv"
if user_reply_std == 'yes':
    latest_file = "standardized_data.csv"
if user_reply_fix_out == 'yes':
    latest_file = "final_cleaned_data.csv"

df = pd.read_csv(latest_file)
print(f"\n✅ Loaded latest data from '{latest_file}'")

col_list = df.columns.tolist()

target_prompt = f"""
You are a data scientist helping a beginner.
The dataset has these columns: {col_list}

Show the column names nicely numbered like:
1. column_name
2. column_name

Then ask them: Which column do you want to predict?
Keep it very simple and friendly.
"""

print("\n🤖 Gemma is looking at your columns...")
response_target = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': target_prompt}])
print("\n" + response_target['message']['content'])

target_col = input("\nEnter the column name you want to predict: ").lower().strip()

while target_col not in df.columns:
    print(f"❌ '{target_col}' not found. Please check the spelling.")
    target_col = input("Enter the column name again: ").lower().strip()

print(f"\n✅ Target column set to: '{target_col}'")

unique_vals = df[target_col].nunique()

if unique_vals <= 10:
    problem_type = "Classification"
else:
    problem_type = "Regression"

problem_prompt = f"""
The user wants to predict '{target_col}'.
It has {unique_vals} unique values.
The problem type is {problem_type}.

Explain in simple words:
- What {problem_type} means in this context
- Why this column is {problem_type}
- What kind of answer the model will give them
One short paragraph. Keep it very simple.
"""

print("\n🤖 Gemma is analyzing your target column...")
response_problem = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': problem_prompt}])
print("\n" + response_problem['message']['content'])
print(f"\n📊 Problem Type: {problem_type}")
print(f"🎯 Target Column: {target_col}")
# --- PHASE 8: FEATURE ENGINEERING ---

cols_to_drop = [col for col in ['employee_id', 'is_missing'] if col in df.columns]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"\n🗑️ Dropped identity/leftover columns: {cols_to_drop}")

col_info = df.dtypes.to_string()
sample = df.head(3).to_string()

feature_prompt = f"""
You are a data scientist. Look at these columns and sample data:
Columns: {col_info}
Sample: {sample}
Target column we are predicting: {target_col}

Suggest 2-3 simple new columns we can create from existing ones that would 
help predict {target_col} better. 

For example:
- if there is age and experience, suggest salary_per_year_experience = salary / experience_years
- if there is performance and training, suggest performance_per_training = performance_score / training_hours

Only suggest things that are mathematically possible with the existing columns.
Keep suggestions simple and explain each one in one line.
"""

print("\n🤖 Gemma is thinking about new features we can engineer...")
response_fe = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': feature_prompt}])
print("\n" + response_f)