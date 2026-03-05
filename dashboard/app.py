import sys
import os

# allow importing project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from llm_toolkit.models.loader import load_model
from llm_toolkit.explainability.attention import extract_attention
from llm_toolkit.explainability.token_importance import get_token_importance
from llm_toolkit.visualization.attention_plot import plot_attention
from llm_toolkit.visualization.token_bar_chart import plot_token_importance


# load model
model, tokenizer = load_model()

st.title("LLM Interpretability Toolkit")

text = st.text_input("Enter text")

if text:

    # extract attention
    attention = extract_attention(model, tokenizer, text)

    # tokenize input
    tokens = tokenizer.tokenize(text)

    st.subheader("Tokens")
    st.write(tokens)

    # layer selector
    layer = st.slider(
        "Select Transformer Layer",
        0,
        len(attention) - 1,
        0
    )

    # head selector
    head = st.slider(
        "Select Attention Head",
        0,
        attention[layer].shape[1] - 1,
        0
    )

    # attention visualization
    st.subheader("Attention Heatmap")

    fig = plot_attention(attention, tokens, layer, head)

    st.pyplot(fig)

    # token importance
    st.subheader("Important Words / Model Reasoning")

    importance = get_token_importance(model, tokenizer, text)

    for token, score in importance:
        st.write(f"{token} : {float(score):.4f}")

    # bar chart visualization
    st.subheader("Token Importance Visualization")

    fig_bar = plot_token_importance(importance)

    st.pyplot(fig_bar)