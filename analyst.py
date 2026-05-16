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
