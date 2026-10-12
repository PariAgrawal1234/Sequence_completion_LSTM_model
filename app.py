
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="LSTM Predictor",
    page_icon="✍️",
    layout="centered"
)

st.title("LSTM Predictor")
st.write("Enter a text sequence and let your trained LSTM predict what comes next.")

@st.cache_resource
def load_components():
    model = tf.keras.models.load_model("model_lstm.h5", compile=False)

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    try:
        with open("max_len.pkl", "rb") as f:
            max_len = int(pickle.load(f))
    except FileNotFoundError:
        max_len = None

    try:
        with open("index_to_word.pkl", "rb") as f:
            index_to_word = pickle.load(f)
    except FileNotFoundError:
        index_to_word = tokenizer.index_word

    model_input_length = model.input_shape[1]

    if model_input_length is not None:
        input_length = int(model_input_length)
    elif max_len is not None:
        input_length = max_len - 1
    else:
        raise ValueError(
            "Cannot determine the model input length. "
            "Please provide max_len.pkl."
        )

    return model, tokenizer, index_to_word, input_length


try:
    model, tokenizer, index_to_word, input_length = load_components()
    st.success("LSTM model loaded successfully!")
except Exception as e:
    st.error(f"Could not load the model or tokenizer: {e}")
    st.stop()


def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return None

    sequence = sequence[-input_length:]

    padded_sequence = pad_sequences(
        [sequence],
        maxlen=input_length,
        padding="pre",
        truncating="pre"
    )

    predictions = model.predict(padded_sequence, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))

    if predicted_index == 0:
        return None

    word = index_to_word.get(predicted_index)

    if word is None:
        return None

    return str(word)

# Streamlit interface
st.subheader("Generate text")

prompt = st.text_area(
    "Enter your starting sequence",
    placeholder="For example: machine learning is",
    height=120
)

num_words = st.radio(
    "Number of words to predict",
    options=[5, 10],
    horizontal=True
)

if st.button("Generate next words", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("Please enter a starting sequence.")
    else:
        generated_text = prompt.strip()
        generated_words = []

        with st.spinner("Generating text..."):
            for _ in range(num_words):
                next_word = predict_next_word(generated_text)

                if next_word is None:
                    st.warning(
                        "Generation stopped because the model returned "
                        "an unknown or padding token."
                    )
                    break

                generated_words.append(next_word)
                generated_text += " " + next_word

        st.subheader("Generated continuation")

        if generated_words:
            st.write("**Original prompt:**")
            st.write(prompt.strip())

            st.write("**Predicted next words:**")
            st.write(" ".join(generated_words))

            st.write("**Complete sequence:**")
            st.info(generated_text)

            st.caption(
                f"Generated {len(generated_words)} of "
                f"{num_words} requested words."
            )
        else:
            st.error(
                "No words could be generated. Check your tokenizer, "
                "model output, and preprocessing."
            )

st.divider()
