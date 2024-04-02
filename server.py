import os
import torch
from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from utils.prepare_vectordb import PrepareVectorDB
from utils.load_config import LoadConfig
from typing import List

app = Flask(__name__)
APPCFG = LoadConfig()

# Create temp_files directory if it doesn't exist
temp_files_dir = "temp_files"
if not os.path.exists(temp_files_dir):
    os.makedirs(temp_files_dir)

# Load the LLM and Tokenizer
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(
    APPCFG.llm_engine, token=APPCFG.gemma_token, device=APPCFG.device
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="google/gemma-7b-it",
    token=APPCFG.gemma_token,
    torch_dtype=torch.float16,
    device_map=APPCFG.device,
)
app_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


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


@app.route("/generate_text", methods=["POST"])
def generate_text():
    data = request.json
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", 1000)
    do_sample = data.get("do_sample", True)
    temperature = data.get("temperature", 0.1)
    top_k = data.get("top_k", 50)
    top_p = data.get("top_p", 0.95)

    tokenized_prompt = app_pipeline.tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    outputs = app_pipeline(
        tokenized_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    return jsonify({"response": outputs[0]["generated_text"][len(tokenized_prompt) :]})


if __name__ == "__main__":
    app.run(debug=False, port=8888)
