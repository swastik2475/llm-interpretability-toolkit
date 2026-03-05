import torch

def token_gradients(model, tokenizer, text):

    inputs = tokenizer(text, return_tensors="pt")

    input_ids = inputs["input_ids"]

    input_ids = input_ids.clone().detach().requires_grad_(True)

    outputs = model(input_ids)

    loss = outputs.logits.sum()

    loss.backward()

    gradients = input_ids.grad

    return gradients