import matplotlib.pyplot as plt

def plot_attention(attention, tokens, layer, head):

    # select layer
    layer_attention = attention[layer]

    # remove batch dimension
    layer_attention = layer_attention[0]

    # select head
    attn = layer_attention[head].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8,8))

    heatmap = ax.imshow(attn, cmap="viridis")

    fig.colorbar(heatmap)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))

    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)

    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")

    ax.set_title(f"Attention Heatmap (Layer {layer}, Head {head})")

    return fig