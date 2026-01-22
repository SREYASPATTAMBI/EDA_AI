import pandas as pd
import ollama

# 1. LOAD THE DATA
# Make sure your file is named 'data.csv' and is in the same folder!
try:
    df = pd.read_csv("data.csv")
    print("✅ Successfully loaded data.csv")
except FileNotFoundError:
    print("❌ Error: data.csv not found. Please add it to the folder.")
    exit()

# 2. EXTRACT THE FACTS (The 'Scout' phase)
# We don't send the whole file, just the "DNA" of the file.
facts = {
    "columns": df.columns.tolist(),
    "rows": len(df),
    "missing_values": df.isnull().sum().to_dict(),
    "sample_data": df.head(3).to_string() # First 3 rows
}

# 3. ASK GEMMA (The 'Brain' phase)
print("🤖 Gemma is analyzing... please wait.")

# We combine our facts into a prompt (instructions)
prompt = f"""
You are a professional data analyst. Use this data metadata:
- Columns: {facts['columns']}
- Total Rows: {facts['rows']}
- Missing Values: {facts['missing_values']}

Please follow this format:
1. DATA STRUCTURE: (Tell me the column count)
2. DATA CLEANLINESS: (List the missing values count per column)
3. WHAT ARE THE TYPE OF MISSING VALUE : (UNDERSTAND AND GIVE WHAT TYPES ARE THE MISSING VALUES MCAR,MAR,MNAR)
3. SUMMARY: (3-sentence overview)
4. RECOMMENDATION: (One analysis suggestion)
"""

response = ollama.chat(
    model='gemma2:2b', 
    messages=[{'role': 'user', 'content': prompt}]
)

# 4. SHOW THE REPORT
print("\n--- ANALYST REPORT ---")
print(response['message']['content'])