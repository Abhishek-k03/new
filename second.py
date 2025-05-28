import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import csv

jobs_df = pd.DataFrame('C:\Users\91879\OneDrive\Desktop\proj\extracted_data\jobs_df.csv')
resume_df = pd.read_csv('/content/drive/MyDrive/resume_data/extracted_features1.csv')  # Replace with your file
jobs_df['skills_str'] = jobs_df['Top Skills'].fillna('').apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

# Build user skills list/string
all_resume_skills = ','.join(resume_df['Technical skills'][-1])
user_skills = [skill.strip() for skill in all_resume_skills.split(',') if skill.strip()]
user_skills_str = ', '.join(user_skills)

# TF-IDF setup
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(', '))
documents = jobs_df['skills_str'].tolist() + [user_skills_str]
tfidf_matrix = vectorizer.fit_transform(documents)

user_vec = tfidf_matrix[-1]
job_vecs = tfidf_matrix[:-1]
similarities = cosine_similarity(user_vec, job_vecs).flatten()
top_indices = similarities.argsort()[-5:][::-1]
top_jobs = jobs_df.iloc[top_indices]

# Build a list of dictionaries for job skills
job_skills_per_job = []
for _, row in jobs_df.iterrows():
    job_dict = {
        "job_title": row['Job Title'],
        "skills": [skill.strip() for skill in row['Top Skills'].split(',') if skill.strip()]
    }
    job_skills_per_job.append(job_dict)

# Your Gemini API key
API_KEY = "YOUR_API_KEY_HERE"
OUTPUT_CSV_PATH = "gemini_skill_gap_analysis.csv"

# Build the prompt
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

def save_response_to_csv(gemini_response, csv_path=OUTPUT_CSV_PATH):
    fieldnames = ["Gemini Skill Gap Analysis"]
    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({fieldnames[0]: gemini_response})
        print(f"CSV file saved to {csv_path}")
    except IOError as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    try:
        gemini_response = get_gemini_response(prompt, API_KEY)
        print(gemini_response)
        save_response_to_csv(gemini_response)
    except Exception as e:
        print(f"An error occurred: {e}")