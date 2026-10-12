# LSTM Next Word Predictor

A simple text generation app that uses a trained LSTM model to predict the next 5 or 10 words from a given text sequence.

## Features
- Predict the next 5 or 10 words.
- Generate text using a trained LSTM model.
- Simple interactive UI built with Streamlit.

## Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- NumPy

## Project Structure
```text
├── app.py
├── model_lstm.h5
├── tokenizer.pkl
├── max_len.pkl
├── index_to_word.pkl
└── requirements.txt
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Enter a starting text sequence, select the number of words, and click **Generate next words**.