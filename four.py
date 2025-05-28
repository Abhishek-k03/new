import pandas as pd
import requests
import json
import csv
import re
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === File Paths ===
RESUME_PATH = r"C:\Users\91879\OneDrive\Desktop\proj\extracted_data\extracted_data_Sample_Resume.csv"
JOBS_PATH = r"C:\Users\91879\OneDrive\Desktop\proj\extracted_data\jobs_df.csv"
OUTPUT_CSV_PATH = "gemini_skill_gap_analysis.csv"

# === Gemini API Key ===
API_KEY = "AIzaSyAQE5qC2pvXkayebW5jVcWLNVW5NSzyCj8"  # Keep private in production

# === Load Data ===
resume_df = pd.read_csv(RESUME_PATH)
jobs_df = pd.read_csv(JOBS_PATH)

if resume_df.empty:
    raise ValueError("Resume DataFrame is empty.")

# === Process Resume Skills ===
all_resume_skills = resume_df['Technical skills'].iloc[-1]  # Get last resume's skills

if isinstance(all_resume_skills, str):
    user_skills = [skill.strip() for skill in all_resume_skills.split(',') if skill.strip()]
else:
    user_skills = [skill.strip() for skill in all_resume_skills if skill.strip()]

user_skills_str = ', '.join(user_skills)
print("User Skills String:", user_skills_str)

# === Safe literal_eval for job skills ===
def safe_literal_eval(x):
    try:
        return literal_eval(x) if isinstance(x, str) else x
    except (ValueError, SyntaxError):
        return []

jobs_df['Top Skills'] = jobs_df['Top Skills'].apply(safe_literal_eval)
jobs_df['skills_str'] = jobs_df['Top Skills'].apply(lambda skills: ', '.join(skills) if isinstance(skills, list) else '')

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(tokenizer=lambda x: re.split(r',\s*', x))
documents = jobs_df['skills_str'].tolist() + [user_skills_str]
tfidf_matrix = vectorizer.fit_transform(documents)

# === Compute Similarities ===
user_vec = tfidf_matrix[-1]
job_vecs = tfidf_matrix[:-1]
similarities = cosine_similarity(user_vec, job_vecs).flatten()
top_indices = similarities.argsort()[-5:][::-1]
top_jobs = jobs_df.iloc[top_indices]

# === Prepare Prompt Data ===
job_skills_per_job = []
for _, row in top_jobs.iterrows():
    job_dict = {
        "job_title": row['Job Title'],
        "skills": [skill.strip() for skill in row['skills_str'].split(',') if skill.strip()]
    }
    job_skills_per_job.append(job_dict)

# === Build Prompt ===
prompt = f"""
You are an expert career coach. Below are two pieces of information:

1. A list of the top 5 job roles and their required skills, formatted as a list of dictionaries:
{json.dumps(job_skills_per_job, indent=2)}

2. A list of the user's skills:
{json.dumps(user_skills, indent=2)}

For each job, do the following:
- List the skills the user has that match the job's requirements ("Matching Skills").
- List the skills required for the job that the user does not have ("Skill Gaps").

Please present your analysis clearly, job by job, using two sections per job: 1. Matching Skills, 2. Skill Gaps.
"""

# === Gemini API Call ===
def get_gemini_response(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [
                {"text": prompt}
            ]
        }]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")
    except (KeyError, IndexError):
        raise Exception("Unexpected response format from Gemini API")

# === Save Gemini Response to CSV ===
def save_response_to_csv(gemini_response, csv_path=OUTPUT_CSV_PATH):
    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Gemini Skill Gap Analysis"])
            for line in gemini_response.strip().split('\n'):
                writer.writerow([line])
        print(f"CSV file saved to {csv_path}")
    except IOError as e:
        print(f"Error saving CSV: {e}")

# === Main Execution ===
if __name__ == "__main__":
    try:
        gemini_response = get_gemini_response(prompt, API_KEY)
        print("\n=== Gemini Skill Gap Analysis ===\n")
        print(gemini_response)
        save_response_to_csv(gemini_response)
    except Exception as e:
        print(f"An error occurred: {e}")
