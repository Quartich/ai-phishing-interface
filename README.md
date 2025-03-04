# AI Phishing Detection Web Interface
Ferris State University half-semester capstone project, March 4, 2025.

Thomas Fairfield, Trenton Mitchell, Garrett Blaylock

At the time of creation, we found Qwen2.5-7B-Instruct-1M-Q5_K_M and kunoichi-dpo-v2-7b.Q4_K_M to be the best performing small deployment models for the task of recognizing phishing.

The neural network we developed utilized an LSTM with two dropout layers to minimize overfitting.  The LSTM model was trained on a synthetic dataset from Kaggle.com featuring around 90,000 emails in total, safe and phishing.  The dataset was not friendly for immediate use and required new column names, and code to allow safe ingestion due to null entries.

token-cap.py is the web interface, which can connect to an LLM api to generate a response. KoboldCPP was the API of choice running locally. The phishingmodel.ipynb Jupyter notebook is used to train the LSTM, using the dataset from: https://www.kaggle.com/datasets/subhajournal/phishingemails

Modifying some hyperparameters, batches, and epochs may improve performance.  The notebook saves both the model as a .h5 and the tokenizer as a .json file.  These are required to use the "Neural Network" option of the web interface.
