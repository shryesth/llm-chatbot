from flask import Flask, request, jsonify
from utils.prepare_vectordb import PrepareVectorDB
from utils.load_config import LoadConfig
from typing import List
import os

app = Flask(__name__)
APPCFG = LoadConfig()

# Create temp_files directory if it doesn't exist
temp_files_dir = "temp_files"
if not os.path.exists(temp_files_dir):
    os.makedirs(temp_files_dir)

@app.route('/add_pdf', methods=['POST'])
def add_pdf():
    # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    # Get the uploaded file
    data = request.files['file']

    # Check if the file name is not empty
    if data.filename == '':
        return jsonify({'error': 'File name is empty'}), 400

    try:
        # Save the file temporarily
        temp_file_path = os.path.join(temp_files_dir, data.filename)
        print("Saving temporary file to:", temp_file_path)
        data.save(temp_file_path)

        # Create an instance of PrepareVectorDB
        prepare_vectordb_instance = PrepareVectorDB(
            data_directory=[temp_file_path],
            persist_directory=APPCFG.custom_persist_directory,
            chunk_size=APPCFG.chunk_size,
            chunk_overlap=APPCFG.chunk_overlap
        )

        # Prepare and save the vector database
        prepare_vectordb_instance.prepare_and_save_vectordb()

        # Delete the temporary file
        os.remove(temp_file_path)

        return jsonify({'message': 'Addition is successful to database'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=8888)
