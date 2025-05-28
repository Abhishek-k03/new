import requests
import fitz  # PyMuPDF
import csv
import json
import os
import re

# Set your Gemini API key here
API_KEY = "AIzaSyAQE5qC2pvXkayebW5jVcWLNVW5NSzyCj8"  # Replace with your actual API key

# Hardcoded CSV output file path
OUTPUT_CSV_PATH = "/content/drive/MyDrive/resume_model/extracted_features1.csv"

# Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

# Send text to Gemini for resume parsing
def ats_extractor_with_gemini(pdf_path):
    resume_text = extract_text_from_pdf(pdf_path)

    prompt = '''
Extract the following details from the resume and return them in valid JSON format. Do not include any markdown, code blocks, or additional text outside the JSON object. The response must be parseable by a JSON parser. Return an object with these keys:
1. employment_details: List of objects with "title" and "company"
2. technical_skills: List of skills
3. soft_skills: List of skills
4. qualification: Highest qualification or education details if available

Example output:
{
  "employment_details": [{"title": "Software Engineer", "company": "Example Corp"}],
  "technical_skills": ["Python", "Java"],
  "soft_skills": ["Communication", "Teamwork"],
  "qualification": "Bachelor's in Computer Science"
}
Resume:
'''

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt + "\n\nResume:\n" + resume_text}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        print("Full API response:", result)  # Debug: Print the full response
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        raise Exception(f"Request failed: {response.status_code}\n{response.text}")

# Parse JSON string safely and return dict or empty dict on error
def parse_json_response(json_str):
    try:
        # Remove markdown code block syntax (```json ... ```) if present
        json_str = re.sub(r'^```json\n|\n```$', '', json_str, flags=re.MULTILINE)
        # Remove any leading/trailing whitespace
        json_str = json_str.strip()
        # Parse the cleaned JSON string
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON response. Error: {e}")
        print("Raw response:")
        print(json_str)
        return {}

# Save extracted info to CSV
def save_to_csv(data, csv_path=OUTPUT_CSV_PATH):
    fieldnames = [
        "Employment details", "Technical skills", "Soft skills", "Qualification"
    ]

    def serialize_field(field):
        if isinstance(field, list):
            if len(field) > 0 and isinstance(field[0], dict):
                return json.dumps(field, ensure_ascii=False)
            return ", ".join(field)
        elif field is None:
            return ""
        else:
            return str(field)

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "Employment details": serialize_field(data.get("employment_details")),
            "Technical skills": serialize_field(data.get("technical_skills")),
            "Soft skills": serialize_field(data.get("soft_skills")),
            "Qualification": serialize_field(data.get("qualification"))
        })

if __name__ == "__main__":
    pdf_resume_path = "/content/drive/MyDrive/resume_model/Sample_Resume.pdf"  # Update this path

    raw_response = ats_extractor_with_gemini(pdf_resume_path)
    parsed_data = parse_json_response(raw_response)

    if parsed_data:
        save_to_csv(parsed_data)
        print(f"Data saved to {OUTPUT_CSV_PATH}")
    else:
        print("No valid data extracted.")
