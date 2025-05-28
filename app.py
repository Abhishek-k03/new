from flask import Flask, request, render_template, send_from_directory, jsonify
import os
from resume_parser import extract_text_from_pdf, ats_extractor_with_gemini, parse_json_response, save_to_csv

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'extracted_data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Process the PDF
            raw_response = ats_extractor_with_gemini(filename)
            parsed_data = parse_json_response(raw_response)

            if parsed_data:
                csv_basename = f"extracted_data_{os.path.splitext(os.path.basename(filename))[0]}.csv"
                csv_filename = os.path.join(app.config['DATA_FOLDER'], csv_basename)
                save_to_csv(parsed_data, csv_filename)
                # Show a page with the download link
                return render_template(
                    "success.html",
                    csv_link=f"/data/{csv_basename}"
                )
            else:
                return jsonify({'error': 'No valid data extracted.'}), 400

    return render_template('upload.html')

@app.route('/data/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['DATA_FOLDER'], filename, as_attachment=True)

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"An error occurred: {e}")
    return jsonify({'error': 'An unexpected error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)