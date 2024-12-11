from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import openai
import os
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)
CORS(app)

openai.api_key = "api_key"

pc = Pinecone(api_key=os.environ.get("api_key"))

index_name = "myindex"  

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

vector_store = pc.Index(index_name)

def generate_embeddings(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response["data"][0]["embedding"]
    except openai.error.OpenAIError as e:
        print(f"Error generating embeddings: {e}")
        return None


def generate_questions(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Generate thoughtful questions based on the following content. "
                           "Ensure the questions are related to the main ideas, concepts, and details of the text provided. "
                           "Avoid focusing on programming unless the text specifically mentions code or programming concepts.\n\n"
                           f"Text:\n{text}"
            }]
        )
        questions = [q.strip() for q in response.choices[0].message['content'].strip().split('\n') if q.strip()]
        return questions
    except openai.error.OpenAIError as e:
        print(f"Error generating questions: {e}")
        return []


def verify_question_quality(question, text):
    rubric = [
        {"criterion": "Clarity", "description": "Is the question clearly written and easy to understand?"},
        {"criterion": "Relevance", "description": "Does the question relate directly to the content of the uploaded material?"},
        {"criterion": "Complexity", "description": "Is the question appropriately complex for the intended audience?"},
        {"criterion": "Correctness", "description": "Does the question accurately reflect the content, without misleading or ambiguous wording?"},
        {"criterion": "Format", "description": "Is the question formatted correctly (multiple-choice, open-ended, code-related)?"}
    ]
    
    rubric_str = "\n".join([f"{rubric_item['criterion']}: {rubric_item['description']}" for rubric_item in rubric])
    prompt = f"""
    You are a quality evaluator for questions based on the content of technical material. Evaluate the following question based on the rubric provided:
    
    Rubric:
    {rubric_str}
    
    Question: {question}
    
    The content of the uploaded material is as follows:
    {text[:500]}... (truncated for context)
    
    Provide a detailed assessment of the question based on the rubric and a final rating (Acceptable/Needs Improvement/Rejected). 
    Respond with the evaluation and reasoning.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        evaluation = response.choices[0].message['content']
        return evaluation
    except openai.error.OpenAIError as e:
        print(f"Error verifying question: {e}")
        return "Error in verification"

def retrieve_relevant_context(query):
    query_embedding = generate_embeddings(query)
    if query_embedding:
      
        results = vector_store.query(
            vector=query_embedding,  
            top_k=5, 
            include_metadata=True  
        )
        context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
        return context
    return ""

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

  
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text() or ""
    if page_text:
        text += page_text
        
        embedding = generate_embeddings(page_text)
        if embedding:
            vector_store.upsert([{
                "id": f"page-{page_num + 1}",  
                "values": embedding,
                "metadata": {"text": page_text}
            }])
    else:
        print(f"Warning: Page {page_num + 1} has no extractable text.")  
    if not text:
        return jsonify({"error": "No extractable text found in the PDF."}), 400

   
    questions = generate_questions(text)

    
    verified_questions = []
    for question in questions:
        evaluation = verify_question_quality(question, text)
        if "Acceptable" in evaluation:
            verified_questions.append(question)

    return jsonify({"questions": verified_questions})

@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    question = data.get("question")
    answer = data.get("answer")

    if not question or not answer:
        return jsonify({"error": "Question or answer missing"}), 400

    context = retrieve_relevant_context(question)

    if not context:
        return jsonify({"error": "No relevant context found"}), 400

    try:
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Using the following context:\n{context}\n\nIs the answer (code or text) '{answer}' correct for the question '{question}'? If the answer is incorrect or needs improvement, please provide suggestions on how to improve it."}
            ]
        )

        verification = response.choices[0].message['content']

      
        return jsonify({"verification": verification})

    except openai.error.OpenAIError as e:
        print(f"Error verifying answer: {e}")
        return jsonify({"error": "Error with OpenAI verification service"}), 500

@app.route('/reset', methods=['POST'])
def reset_index():
    try:
        Pinecone.delete_index(index_name)
        Pinecone.create_index(index_name, dimension=1536)
        global vector_store
        vector_store = Pinecone.Index(index_name)
        return jsonify({"message": "Index has been reset."})
    except Pinecone.exceptions.PineconeException as e:
        print(f"Error resetting index: {e}")
        return jsonify({"error": "Error resetting Pinecone index"}), 500

if __name__ == '__main__':
    app.run(port=5001)
