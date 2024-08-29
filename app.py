import pandas as pd
import joblib
from flask import Flask, request, jsonify
import google.generativeai as palm
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
palm_api_key = os.getenv("PALM_API_KEY")

# Configure with your API key
palm.configure(api_key=palm_api_key)

# Retrieve and select a generative AI model that supports 'generateText'
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
if models:
    model = models[0].name
else:
    model = None

app = Flask(__name__)

# Initialize a dictionary to store previous chats
previous_chats = {}

@app.route('/medical', methods=['POST'])
def medical_question():
    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        question = input_data.get('question', '')

        if not model:
            return jsonify({"error": "No suitable AI model found"}), 500

        # Check if the question is a greeting or unrelated to health
        if "greeting" in question.lower() or "who are you" in question.lower():
            response = "painvestex is an assistant here to help with any questions you have."
        else:
            # Check if there are previous chats related to this question
            related_chat = previous_chats.get(question.lower())
            if related_chat:
                response = related_chat
            else:
                general_prompt = f"""
                You are painvestex, a highly knowledgeable assistant capable of answering questions across a wide range of topics.
                Your goal is to provide clear, concise, and accurate responses to the user's query. 
                If the user asks a question that is out of the ordinary or needs a brief conclusion, summarize the main points and give a brief, insightful explanation.
                The question: {question}
                """
                response = palm.generate_text(
                    model=model,
                    prompt=general_prompt,
                    max_output_tokens=800,
                ).result

                # Store the response in the previous chats dictionary
                previous_chats[question.lower()] = response

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=4000)
