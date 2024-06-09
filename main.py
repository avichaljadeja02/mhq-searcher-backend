from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import chromadb
import PyPDF2
from openai import OpenAI
client = OpenAI()




import pandas as pd
import time

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

chroma_client = chromadb.PersistentClient(path="/chromaDB")
collection = chroma_client.get_or_create_collection(name="mhq_docs")

@app.route('/api/drop', methods=['POST'])
def drop_chroma():
    chroma_client.delete_collection(name="mhq_docs")
    return jsonify({'message': 'ChromaDB collection dropped!'}), 200

def validate_result(question, generated_response):
    prompt = f"""
    You are working as a validation agent for a financial tech project: 
    If the answer is not correct - do not mention that. Instead ONLY generate an answer and reply to the question!!!!
    Please ensure that every answer you give is the same as before!
    Your job is to validate if the answer is contained within the document and makes sense to the best of your ability!!
    If it does not - generate the answer for the question and return only that!! to the best of your knowledge! At the end add - powered by CHATGPT
    If it does - please say "answer is correct!". 
    Additionally return the part of the response that is associated with the question and supproting/additional details 
    you find might be helpful to support the answer!!
    For the following question - {question} 
    We got the following response from a document: {generated_response["documents"][0]}
    Do not mention anything about the document or anything related to that!
    You need to be 100% accurate as this is being implemented in an extremely impactful process.
    Always ensure your response is readable to an end user and can simply be displayed in the project!!
    """


    try:
        print("Going to GPT to validate")
        response = client.chat.completions.create(
            model="gpt-4o",
                messages=[
                {"role": "system", "content": "You are a highly accurate validation agent."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=1.2,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.7,
            stop=['Human:', 'AI:']
        )
        res = response.choices[0].message.content.lower()
        print(res)
        if "answer is correct!" in res:
            return generated_response["documents"][0][0]
        else: 
            return res
    except Exception as e:
        return f"Error during validation: {str(e)}"

# Example usage within your Flask endpoint
@app.route('/api/query', methods=['POST'])
def search_chroma():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'Query parameter is missing'}), 400

    try:
        results = collection.query(
            query_texts=[query],
            n_results=1
        )

        if results:
            validated_result = validate_result(query, results)
            return jsonify({'validated_result': validated_result}), 200
        else:
            return jsonify({'error': 'No results found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            document_text = ''
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                document_text += page.extract_text()

        document_id = f"doc_{os.path.splitext(filename)[0]}"

        collection.upsert(
            documents=[document_text],
            ids=[document_id]
        )
        print("added document with document text :- \n", document_text)
        os.remove(file_path)

        return jsonify({'message': 'File processed and data stored in ChromaDB', 'documentId': document_id})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
