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