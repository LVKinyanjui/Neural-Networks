# html_loader.py
import os
from bs4 import BeautifulSoup

def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text()

    return text_content

def load_html_files(directory_path):
    data = []

    # Iterate through files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.html'):
            file_path = os.path.join(directory_path, filename)

            # Extract text from HTML and append to the data list
            text_content = extract_text_from_html(file_path)
            data.append(text_content)

    return data
