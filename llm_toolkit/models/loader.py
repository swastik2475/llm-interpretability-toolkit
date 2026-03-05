from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        output_attentions=True,
        output_hidden_states=True,
        device_map=None
    )

    model.eval()

    return model, tokenizer