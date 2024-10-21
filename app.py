from flask import Flask, request, jsonify, render_template
from src.retrieval import retrieve_similar_ui
from src.generation import generate_image_from_prompt
import os

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Renders form for input

# Route for generating the UI design based on user input
@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.form['query']
    
    similar_uis = retrieve_similar_ui(user_input)
    
    prompt = similar_uis[0]['description']
    generated_image = generate_image_from_prompt(prompt)
    
    image_path = "static/generated_ui.png"
    generated_image.save(image_path)

    return render_template('result.html', query=user_input, image_path=image_path, similar_uis=similar_uis)

if __name__ == '__main__':
    app.run(debug=True)
