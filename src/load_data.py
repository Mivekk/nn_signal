import os
import json
import requests
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_json_files(url, local_dir):
    # Create local directory if it does not exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Get the webpage content
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links ending with .json
    json_links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True) if a['href'].endswith('.json')]

    # Download each .json file
    for link in json_links:
        file_name = os.path.join(local_dir, os.path.basename(link))
        if os.path.exists(file_name):
            # print(f"File already exists: {file_name}")
            continue
        with requests.get(link, stream=True) as r:
            r.raise_for_status()
            with open(file_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        print(f"Downloaded: {file_name}")
        
# Usage
url = 'http://titi.etsii.urjc.es/splendid/samples/'
local_dir = './data/'
print("Downloading json files...")
download_json_files(url, local_dir)

def get_json_files_content(local_dir):
    # List all .json files in the local directory
    json_contents = []
    for file_name in os.listdir(local_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(local_dir, file_name)
            # Open and save the content of each .json file
            with open(file_path, 'r') as f:
                content = json.load(f)
                
                json_contents.append(content)
    
    return np.array(json_contents)

json_contents = get_json_files_content(local_dir)

def extract_interior_exterior_peaks(json_contents):
    interior_elements = []
    exterior_elements = []
    peaks_elements = []
    for file_content in json_contents:
        for entry in file_content:
            interior_elements.append(entry["interior"])
            exterior_elements.append(entry["exterior"])
            peaks_elements.append(entry["peaks"]["serie"])
            
    # Stack interior and exterior to shape (num_entries, sequence_length, num_features)
    combined_elements = np.stack([interior_elements, exterior_elements], axis=-1)
    
    return combined_elements, np.array(peaks_elements)

data, targets = extract_interior_exterior_peaks(json_contents)