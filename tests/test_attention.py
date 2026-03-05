from llm_toolkit.models.loader import load_model
from llm_toolkit.explainability.attention import extract_attention


def test_attention():

    model, tokenizer = load_model()

    text = "The cat chased the mouse"

    attention = extract_attention(model, tokenizer, text)

    assert attention is not None