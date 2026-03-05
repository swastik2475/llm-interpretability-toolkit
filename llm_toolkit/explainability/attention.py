def extract_attention(model, tokenizer, text):

    inputs = tokenizer(text, return_tensors="pt")

    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    outputs = model(**inputs)

    attentions = outputs.attentions

    return attentions