import torch
from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from dotenv import load_dotenv
import os


load_dotenv()


app = Flask(__name__)

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model_id = "gg-hf/gemma-2b-it"
hf_token = os.getenv("GEMMA_TOKEN")
dtype = torch.bfloat16


# Load the LLM and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    device_map="cpu",
    torch_dtype=dtype,
)

@app.route("/generate_text", methods=["POST"])
def generate_Text():
    data = request.json
    chat = data.get("prompt", "")

    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")


    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=False, port=8888)
