import shap

def shap_explain(model, tokenizer, text):

    explainer = shap.Explainer(model)

    shap_values = explainer([text])

    return shap_values