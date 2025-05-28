from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import extract_text_from_pdf, ats_extractor_with_gemini, parse_json_response, save_to_csv

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER   = 'extracted_data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER

# ----------------------------------------------------------------
# 1) Load & vectorize job‐postings once at startup
# ----------------------------------------------------------------
jobs_df = pd.read_csv(os.path.join(DATA_FOLDER, 'jobs_df.csv'))  # adjust path if needed
jobs_df['Top Skills']  = jobs_df['Top Skills'].fillna('').apply(lambda x: x.split(',') if x else [])
jobs_df['skills_str']  = jobs_df['Top Skills'].apply(lambda skills: ', '.join([s.strip() for s in skills]))

tfidf_vectorizer = TfidfVectorizer()
job_skill_matrix   = tfidf_vectorizer.fit_transform(jobs_df['skills_str'])


# ----------------------------------------------------------------
# 2) Define helper to get top-N matches
# ----------------------------------------------------------------
def get_top_job_matches(resume_df, top_n=5):
    # assume resume_df has a column 'Technical skills' in first row
    raw = resume_df.loc[0, 'Technical skills']
    user_skills = [s.strip() for s in raw.split(',') if s.strip()]
    user_skills_str = ', '.join(user_skills)

    user_vec = tfidf_vectorizer.transform([user_skills_str])
    sims     = cosine_similarity(user_vec, job_skill_matrix).flatten()
    top_idxs = sims.argsort()[-top_n:][::-1]

    results = jobs_df.iloc[top_idxs].copy()
    results['match_score'] = sims[top_idxs]
    return results.reset_index(drop=True)


# ----------------------------------------------------------------
# 3) Upload & parse endpoint, now with matching
# ----------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400

        # save to disk
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)

        # run your existing parser
        raw_response = ats_extractor_with_gemini(input_path)
        parsed_data  = parse_json_response(raw_response)

        if not parsed_data:
            return jsonify({'error': 'No valid data extracted.'}), 400

        # save parsed data to CSV
        base = os.path.splitext(os.path.basename(file.filename))[0]
        csv_name = f"extracted_data_{base}.csv"
        csv_path = os.path.join(app.config['DATA_FOLDER'], csv_name)
        save_to_csv(parsed_data, csv_path)

        # --- NEW: load the resume CSV and run matcher ---
        resume_df  = pd.read_csv(csv_path)
        top_matches = get_top_job_matches(resume_df, top_n=5)

        # render a template that shows the CSV download link + the top matches
        return render_template(
            "success.html",
            csv_link=f"/data/{csv_name}",
            matches=top_matches.to_dict(orient='records')
        )

    return render_template('upload.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/data/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['DATA_FOLDER'], filename, as_attachment=True)


@app.errorhandler(Exception)
def handle_exception(e):
    print(f"An error occurred: {e}")
    return jsonify({'error': 'An unexpected error occurred.'}), 500


if __name__ == '__main__':
    app.run(debug=True)