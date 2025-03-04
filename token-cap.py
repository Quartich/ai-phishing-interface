import gradio as gr
import requests
import json
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

#global vars for nn stuff
model = None
model_loaded = False
tokenizer = None
tokenizer_loaded = False

max_vocab_size = 20000
max_length = 500

#this loads the lstm only on demand
def load_model():
    global model, model_loaded
    if not model_loaded:
        try:
            if os.path.exists("phishing_model.h5"):
                model = tf.keras.models.load_model("phishing_model.h5")
                model_loaded = True
                return True
            else:
                return False
        except Exception as e:  #if they dont have model file
            print(f"Error loading model: {e}")
            return False
    return True

def load_tokenizer():  #tokenizer on demand too
    global tokenizer, tokenizer_loaded
    if not tokenizer_loaded:
        try:
            if os.path.exists("tokenizer.json"):
                with open("tokenizer.json", "r") as f:
                    tokenizer_json = f.read()
                    tokenizer = tokenizer_from_json(tokenizer_json)
                tokenizer_loaded = True
                return True
            else:
                return False
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return False
    return True

def preprocess_text(email, subject, body): #tokenize text
    combined_text = f"{email} {subject} {body}".lower()
    sequence = tokenizer.texts_to_sequences([combined_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    return np.array(padded_sequence)

def check_phishing(email, subject, body, use_llm, use_nn, api_url):
    llm_response = ""
    phishing_probability = 0
    if not email and not subject and not body:
        return "Please provide email content to analyze.", "N/A"
    if use_llm: #llm selected
        try:
            prompt_text = (
                "You are a security expert analyzing emails for phishing attempts.\n"
                "Analyze the following email and explain why it might or might not be a phishing attempt based on common signs.\n\n"
                f"From: {email}\n"
                f"Subject: {subject}\n"
                f"Body: {body}\n\n"
            )#prompt for ai with email info
            payload = {
                "prompt": prompt_text,
                "max_context_length": 2048,
                "max_length": 500,
                "temperature": 0.9,
                "top_p": 0.9,
                "top_k": 100,
                "rep_pen": 1.1,
                "rep_pen_range": 256,
                "rep_pen_slope": 1,
                "typical": 1,
                "tfs": 1,
                "top_a": 0,
                "quiet": False
            }#model parameters
            headers = {"Content-Type": "application/json"}
            response = requests.post(f"{api_url}/api/v1/generate", json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            if "results" in result and len(result["results"]) > 0:
                llm_response = result["results"][0].get("text", "No response received from API")
            else:
                llm_response = "No valid response received from API"  
        except requests.exceptions.RequestException as e:
            llm_response = f"API Error: {str(e)}"
    
    elif use_nn:  #nn
        if load_model() and load_tokenizer():
            features = preprocess_text(email, subject, body)
            try:
                prediction = model.predict(features)[0][0]
                phishing_probability = float(prediction) * 100
                llm_response = "Neural network analysis complete. See probability below."
            except Exception as e:
                llm_response = f"Error during neural network analysis: {str(e)}"
                phishing_probability = "Error"
        else:
            llm_response = "Error: Could not load the neural network model or tokenizer."
            phishing_probability = "N/A"
    else:
        llm_response = "Please select either LLM Model or Neural Network for analysis."
    probability_display = f"{phishing_probability:.2f}%" if isinstance(phishing_probability, (int, float)) else str(phishing_probability)
    return llm_response, probability_display

def update_checkboxes(llm_checked, nn_checked):
    """Ensure only one checkbox is selected at a time"""
    if llm_checked:
        return True, False
    elif nn_checked:
        return False, True
    return False, False

with gr.Blocks(title="Phishing Email Detection System") as app:
    gr.Markdown("# Phishing Email Detection System")
    gr.Markdown("Enter email details below to analyze for phishing attempts")
    
    with gr.Row():
        with gr.Column():
            email_input = gr.Textbox(label="Email Sender")
            subject_input = gr.Textbox(label="Email Subject")
            body_input = gr.Textbox(label="Email Body", lines=15)
            
            with gr.Row():
                llm_checkbox = gr.Checkbox(label="LLM Model", value=True)
                nn_checkbox = gr.Checkbox(label="Neural Network", value=False)
            
            llm_checkbox.change(fn=update_checkboxes, inputs=[llm_checkbox, nn_checkbox], outputs=[llm_checkbox, nn_checkbox])
            nn_checkbox.change(fn=update_checkboxes, inputs=[nn_checkbox, llm_checkbox], outputs=[nn_checkbox, llm_checkbox])
            
            api_input = gr.Textbox(label="LLM API URL", value="http://127.0.0.1:5001")
            submit_button = gr.Button("Click to Check")
        
        with gr.Column():
            result_output = gr.Textbox(label="Analysis Results", lines=30)
            probability_output = gr.Textbox(label="Phishing Probability (Neural Network)")

    submit_button.click(
        fn=check_phishing,
        inputs=[email_input, subject_input, body_input, llm_checkbox, nn_checkbox, api_input],
        outputs=[result_output, probability_output]
    )

if __name__ == "__main__":
    app.launch()
