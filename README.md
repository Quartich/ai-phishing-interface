# ai-phishing-interface
Ferris State University half-semester capstone project, March 4, 2025.

Thomas Fairfield, Trenton Mitchell, Garrett Blaylock

At the time of creation, we found Qwen2.5-7B-Instruct-1M-Q5_K_M and kunoichi-dpo-v2-7b.Q4_K_M to be the best performing small deployment models for the task of recognizing phishing.

The neural network we developed utilized an LSTM with two dropout layers to minimize overfitting.  The LSTM model was trained on a synthetic dataset from Kaggle.com featuring around 90,000 emails in total, safe and phishing.  The dataset was not friendly for immediate use and required new column names, and code to allow safe ingestion due to null entries.
