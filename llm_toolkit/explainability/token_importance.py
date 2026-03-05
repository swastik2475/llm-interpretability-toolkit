import torch

def get_token_importance(model, tokenizer, text):

    inputs = tokenizer(text, return_tensors="pt")

    input_ids = inputs["input_ids"]

    # get embeddings instead of ids
    embeddings = model.transformer.wte(input_ids)

    embeddings = embeddings.clone().detach().requires_grad_(True)

    outputs = model(inputs_embeds=embeddings)

    logits = outputs.logits

    score = logits.sum()

    score.backward()

    grads = embeddings.grad

    # importance score
    importance = grads.abs().sum(dim=-1).squeeze()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

    token_importance = list(zip(tokens, importance.detach().numpy()))

    return token_importance