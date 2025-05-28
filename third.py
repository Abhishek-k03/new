import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
resume_df = pd.read_csv(r"C:\Users\91879\OneDrive\Desktop\proj\extracted_data\extracted_data_Sample_Resume.csv")
jobs_df = pd.read_csv(r"C:\Users\91879\OneDrive\Desktop\proj\extracted_data\jobs_df.csv")

# Process 'Top Skills'
jobs_df['Top Skills'] = jobs_df['Top Skills'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
jobs_df['skills_str'] = jobs_df['Top Skills'].apply(lambda skills: ', '.join(skills) if isinstance(skills, list) else '')

# Process user skills
all_resume_skills = resume_df['Technical skills'].iloc[-1]
user_skills = [skill.strip() for skill in all_resume_skills.split(',') if skill.strip()]
user_skills_str = ', '.join(user_skills)

# Combine job and user skills into a single list of strings
combined_skills = jobs_df['skills_str'].tolist() + [user_skills_str]

# Vectorize using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_skills)

# Separate the job skills and user skills TF-IDF representations
user_vec = tfidf_matrix[-1]
job_vecs = tfidf_matrix[:-1]
similarities = cosine_similarity(user_vec, job_vecs).flatten()
top_indices = similarities.argsort()[-5:][::-1]
top_jobs = jobs_df.iloc[top_indices]

# Display top job matches
print("Top Job Matches:")
print(top_jobs)