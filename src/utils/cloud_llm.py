from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Define upload folder and allowed extensions
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    with open(pdf_docs, 'rb') as f:
        pdf = PdfReader(f)
        number_of_pages = len(pdf.pages)
        # Loop through all pages and extract text
        all_page_text = ""
        for page_num in range(number_of_pages):
            page = pdf.pages[page_num]
            page_text = page.extract_text()  # Extract text from each page
            all_page_text += page_text  # Combine text from all pages
            # Print the extracted text from all pages
        print(all_page_text)
    return all_page_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    safety_settings={
        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
}


    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3, safety_settings= safety_settings)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt )

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization =True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    return(response)
    
@app.route('/')
def index():
    return jsonify({"/upload": "For uploading files", "/generateanswer": "To get the answer from the PDF"})

@app.route('/upload', methods=['POST'])
def upload_file():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
    
    files = request.files.getlist('file')

    pdf_docs = []
    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(file_path)
            pdf_docs.append(file_path)


    if not pdf_docs:
        return jsonify({'error': 'Unsupported file format hence No valid files uploaded'}), 400


    print(pdf_docs[0])
    raw_text = get_pdf_text(pdf_docs[0])
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    
    return jsonify({'message': 'File uploaded and processed successfully'}), 201

@app.route('/generateanswer', methods=['POST'])
def generate_answer():
    # Get user input from JSON request
    user_input_json = request.get_json()
    if user_input_json is None or "userInput" not in user_input_json:
        return jsonify({'error': 'Invalid JSON format or missing "userInput" key'}), 400

    input = user_input_json["userInput"]
    if not input.strip():
        return jsonify({'error': 'User input is empty'}), 400

    answer = user_input(input)

    # Return JSON response
    return jsonify({"answer": answer['output_text']})




if __name__ == '__main__':
    app.run(debug=True)