def extract_hidden_states(model, tokenizer, text):

    inputs = tokenizer(text, return_tensors="pt")

    outputs = model(**inputs)

    hidden_states = outputs.hidden_states

    return hidden_states