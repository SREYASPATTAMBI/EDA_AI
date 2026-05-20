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
# --- PHASE 9: MODEL SELECTION + CROSS VALIDATION ---
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

cv_intro_prompt = f"""
The user is a beginner. Explain in very simple words:
- The target column '{target_col}' has text values like 'yes' and 'no'
- We need to convert them to numbers (yes=1, no=0) so the model can understand
- Then we will test 5 different models on this data
- Each model will be tested using Cross Validation to find the most accurate one
Keep it very simple, 2-3 lines only.
"""

print("\n🤖 Gemma is preparing for model testing...")
response_cv_intro = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': cv_intro_prompt}])
print("\n" + response_cv_intro['message']['content'])

X = df.drop(columns=[target_col])
y = df[target_col].astype(str).str.strip().str.lower()
y = y.replace({'nan': 'no'})

le = LabelEncoder()
y = le.fit_transform(y)
print(f"\n✅ Target column '{target_col}' converted to numbers.")

cat_cols_x = X.select_dtypes(include=['object', 'string']).columns.tolist()
for col in cat_cols_x:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
print(f"✅ Encoded {len(cat_cols_x)} categorical columns: {cat_cols_x}")

if problem_type == "Classification":
    models = {
        "1. Logistic Regression"         : LogisticRegression(max_iter=1000),
        "2. Random Forest"               : RandomForestClassifier(n_estimators=100, random_state=42),
        "3. SVM"                         : SVC(kernel='rbf'),
        "4. KNN"                         : KNeighborsClassifier(n_neighbors=5),
        "5. XGBoost (Gradient Boosting)" : GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    scoring = 'accuracy'
else:
    models = {
        "1. Linear Regression"           : LinearRegression(),
        "2. Random Forest"               : RandomForestRegressor(n_estimators=100, random_state=42),
        "3. SVR"                         : SVR(kernel='rbf'),
        "4. KNN"                         : KNeighborsRegressor(n_neighbors=5),
        "5. XGBoost (Gradient Boosting)" : GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    scoring = 'r2'

print(f"\n🤖 Gemma is running Cross Validation on all 5 models...")
print("-" * 45)

cv_scores = {}

for model_name, model in models.items():
    if "Logistic" in model_name or "SVM" in model_name or "SVR" in model_name or "KNN" in model_name:
        scaler = StandardScaler()
        X_prepared = scaler.fit_transform(X)
    else:
        X_prepared = X.values

    scores = cross_val_score(model, X_prepared, y, cv=5, scoring=scoring)
    avg_score = round(scores.mean() * 100, 2)
    cv_scores[model_name] = avg_score
    print(f"{model_name}: {avg_score}%")

print("-" * 45)
print("✅ Cross Validation complete!")

best_model_name = max(cv_scores, key=cv_scores.get)
best_score = cv_scores[best_model_name]

recommend_prompt = f"""
You are a data scientist explaining results to a beginner.
Here are the cross validation scores for 5 models:
{cv_scores}

The best model is: {best_model_name} with {best_score}% score.
The problem type is: {problem_type}
The dataset has {df.shape[0]} rows and {df.shape[1]} columns.

Explain in simple words:
- What these scores mean
- Why {best_model_name} is the best choice
- One sentence about what makes it better than the others
Keep it simple and friendly, 3-4 lines only.
"""

print("\n🤖 Gemma is analyzing the results...")
response_recommend = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': recommend_prompt}])
print("\n" + response_recommend['message']['content'])

print("\n📊 Here are all your options:")
print("-" * 45)
for i, (name, score) in enumerate(cv_scores.items(), 1):
    marker = "⭐ RECOMMENDED" if name == best_model_name else ""
    print(f"{name}: {score}% {marker}")
print("-" * 45)

user_model_choice = input(f"\nPress ENTER to go with '{best_model_name}' or type a number (1-5) to pick manually: ").strip()

if user_model_choice == "":
    final_model_name = best_model_name
    final_model = models[best_model_name]
    print(f"\n✅ Going with: {final_model_name}")
elif user_model_choice in ["1", "2", "3", "4", "5"]:
    final_model_name = list(models.keys())[int(user_model_choice) - 1]
    final_model = models[final_model_name]
    print(f"\n✅ You picked: {final_model_name}")
else:
    final_model_name = best_model_name
    final_model = models[best_model_name]
    print(f"\n✅ Going with recommended: {final_model_name}")

print(f"\n🎯 Final Model: {final_model_name}")
print(f"📊 CV Score: {cv_scores[final_model_name]}%")
# --- PHASE 10: FEATURE SELECTION + FINAL MODEL TRAINING ---
import pickle
from sklearn.metrics import accuracy_score

feature_sel_prompt = f"""
The user is a beginner. Explain in very simple words:
- We just found that {final_model_name} is the best model
- Now we want to remove columns that are not useful for this model
- This is called Feature Selection
- We check how strongly each column is related to '{target_col}'
- Weak columns get removed so the model can focus on what matters
Keep it very simple, 2-3 lines only.
"""

print("\n🤖 Gemma is explaining feature selection...")
response_fs = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': feature_sel_prompt}])
print("\n" + response_fs['message']['content'])

print("\n🔍 Checking each column's strength with the target...")
print("-" * 45)

temp_df = X.copy()
temp_df['target'] = y

correlations = temp_df.corr()['target'].drop('target').abs()
correlations = correlations.sort_values(ascending=False)

for col, score in correlations.items():
    strength = "✅ Strong" if score >= 0.05 else "❌ Weak"
    print(f"{col}: {round(score, 4)} → {strength}")

print("-" * 45)

threshold = 0.05
selected_features = correlations[correlations >= threshold].index.tolist()
dropped_features = correlations[correlations < threshold].index.tolist()

print(f"\n✅ Keeping {len(selected_features)} strong columns: {selected_features}")
print(f"🗑️ Dropping {len(dropped_features)} weak columns: {dropped_features}")

explain_sel_prompt = f"""
You are a data scientist explaining to a beginner.
We kept these columns for the model: {selected_features}
We dropped these weak columns: {dropped_features}
The model we are using is: {final_model_name}

Explain in simple words why keeping only strong columns helps the model.
2-3 lines only. Keep it very friendly.
"""

print("\n🤖 Gemma is explaining the feature selection results...")
response_fs2 = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': explain_sel_prompt}])
print("\n" + response_fs2['message']['content'])

user_reply_fs = input("\nShould I train the final model with these selected features? (yes/no): ").lower().strip()

if user_reply_fs == 'yes':
    print(f"\n⚙️ Training {final_model_name} on selected features...")

    X_final = X[selected_features]

    if "Logistic" in final_model_name or "SVM" in final_model_name or "SVR" in final_model_name or "KNN" in final_model_name:
        scaler = StandardScaler()
        X_final_scaled = scaler.fit_transform(X_final)
        pickle.dump(scaler, open("scaler.pkl", "wb"))
        print("✅ Data scaled for model.")
    else:
        X_final_scaled = X_final.values
        scaler = None
        print("✅ No scaling needed for this model.")

    final_model.fit(X_final_scaled, y)
    print(f"✅ {final_model_name} trained successfully!")

    train_preds = final_model.predict(X_final_scaled)
    train_acc = round(accuracy_score(y, train_preds) * 100, 2)
    print(f"\n📊 Training Accuracy: {train_acc}%")
    print(f"📊 Cross Validation Score was: {cv_scores[final_model_name]}%")

    final_result_prompt = f"""
    You are a data scientist explaining results to a beginner.
    The model is: {final_model_name}
    Cross Validation Score: {cv_scores[final_model_name]}%
    Training Accuracy: {train_acc}%
    Target column: {target_col}
    Selected features used: {selected_features}

    Explain in simple words:
    - What the training accuracy means
    - Why CV score is more reliable than training accuracy
    - What this model can now do
    Keep it very simple and friendly. 3-4 lines only.
    """

    print("\n🤖 Gemma is explaining the final results...")
    response_final = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': final_result_prompt}])
    print("\n" + response_final['message']['content'])

    pickle.dump(final_model, open("model.pkl", "wb"))
    pickle.dump(selected_features, open("selected_features.pkl", "wb"))
    print(f"\n💾 Model saved to 'model.pkl'")
    print(f"💾 Selected features saved to 'selected_features.pkl'")

    print("\n" + "=" * 45)
    print("🎉 YOUR AI EDA ASSISTANT IS COMPLETE!")
    print("=" * 45)
    print(f"✅ Problem Type     : {problem_type}")
    print(f"✅ Target Column    : {target_col}")
    print(f"✅ Best Model       : {final_model_name}")
    print(f"✅ CV Score         : {cv_scores[final_model_name]}%")
    print(f"✅ Training Accuracy: {train_acc}%")
    print(f"✅ Features Used    : {len(selected_features)}")
    print(f"✅ Model saved to   : model.pkl")
    print("=" * 45)

else:
    print("\n👍 Skipping final training. Your pipeline is complete up to model selection!")