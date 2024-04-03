# Internship Project: LLM-based Chatbot

This repository contains the code for an internship project that implements a large language model (LLM) chatbot.

## Setup

### Clone the Repository:

```bash
git clone https://github.com/shryesth/llm-chatbot.git
```

## Create a .ENV File:

1. Create a file named `.env` in the root directory of the project.
2. Add the following lines to the `.env` file, replacing the placeholders with your actual API keys:

```bash
GEMMA_TOKEN=xxxxxxxxxx
GOOGLE_API_KEY=xxxxxxxxxx
```
## Install Dependencies:
```bash
pip install -r requirements.txt
```

## Run the Server
```bash
python src/runner.py
```

## Additional Notes:
- Due to GPU limitations, the project wasn't able to fully test the GEMMA local LLM. However, responses were still obtained.
- Make a GET request on server to obtain endpoints.
