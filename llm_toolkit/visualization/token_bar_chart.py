import matplotlib.pyplot as plt

def plot_token_importance(token_scores):

    tokens = [t[0] for t in token_scores]
    scores = [t[1] for t in token_scores]

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(tokens, scores)

    ax.set_xlabel("Tokens")
    ax.set_ylabel("Importance Score")

    ax.set_title("Token Importance Visualization")

    plt.xticks(rotation=45)

    return fig