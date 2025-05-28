import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

def generate_top_skills_by_job_title(csv_path):
    # Load and sample the dataset
    df = pd.read_csv(csv_path, usecols=['Job Title', 'skills'], dtype={'skills': 'str'})
    df_sample = df.sample(frac=0.05, random_state=42)  # Or use all data, as needed

    def process_skills(skills_str):
        if pd.isna(skills_str):
            return []
        return [skill.strip().lower() for skill in skills_str.split(',')]

    df_sample['Processed_Skills'] = df_sample['skills'].apply(process_skills)
    df_sample['Skills_Text'] = df_sample['Processed_Skills'].apply(lambda x: ' '.join(x))

    # Vectorize skills with a reduced feature limit
    vectorizer = CountVectorizer(max_features=500)
    X = vectorizer.fit_transform(df_sample['Skills_Text'])

    # Dimensionality reduction
    pca = PCA(n_components=20)
    X_reduced = pca.fit_transform(X.toarray())

    # Clustering
    num_clusters = 147
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
    df_sample['Cluster'] = kmeans.fit_predict(X_reduced)

    # Helper functions
    def get_top_skills(indices):
        all_skills = sum((df_sample.iloc[idx]['Processed_Skills'] for idx in indices if idx < len(df_sample)), [])
        skill_counts = pd.Series(all_skills).value_counts()
        return skill_counts.index.tolist()[:5]  # top 5 skills

    def get_representative_job_title(indices):
        job_titles = df_sample.iloc[list(indices)]['Job Title']
        most_common_title = job_titles.mode()[0]
        return most_common_title

    # Retrieve common skills and job titles
    common_skills = df_sample.groupby('Cluster').indices
    top_skills_by_cluster = {
        get_representative_job_title(indices): get_top_skills(indices)
        for indices in common_skills.values()
    }
    return top_skills_by_cluster