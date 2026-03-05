# LLM Interpretability Toolkit

LLM Interpretability Toolkit is an open-source project that explores how transformer-based language models process and understand text. Large language models are often treated as black boxes, and this project attempts to make their internal behavior more transparent by visualizing attention patterns and token importance.

The toolkit provides a simple interactive dashboard where users can input text and observe how the model distributes attention across tokens. By examining these patterns, users can gain insight into how the model interprets relationships between words and builds contextual understanding.

---

## Project Overview

Transformer-based models rely heavily on attention mechanisms to process language. These mechanisms determine how strongly each token in a sentence interacts with other tokens. By extracting and visualizing these attention weights, we can better understand how the model forms contextual relationships.

This project focuses on three main aspects of interpretability:

1. Visualizing token-level attention between words.
2. Exploring individual transformer layers and attention heads.
3. Identifying which tokens influence the model's internal reasoning.

The project includes an interactive interface that allows users to experiment with different layers and attention heads to observe how attention distributions change within the model.

---

## Features

* Attention heatmap visualization showing relationships between tokens
* Exploration of individual transformer layers
* Exploration of individual attention heads
* Token importance analysis using gradient-based methods
* Interactive Streamlit dashboard for experimentation
* Automated testing and CI pipeline integration

---

## Project Structure

```
llm-interpretability-toolkit
│
├── dashboard
│   └── app.py
│
├── llm_toolkit
│   ├── models
│   │   └── loader.py
│   │
│   ├── explainability
│   │   ├── attention.py
│   │   ├── gradients.py
│   │   └── shap_explainer.py
│   │
│   └── visualization
│       ├── attention_plot.py
│       └── token_bar_chart.py
│
├── tests
│   └── test_attention.py
│
├── .github
│   └── workflows
│       └── ci.yml
│
└── requirements.txt
```

---

## Screenshots

### Dashboard Interface

This view shows the main interface where the user enters text and observes how the model tokenizes the sentence.

![Dashboard](Screenshot%202026-03-05%20212050.png)

---

### Attention Heatmap

The attention heatmap displays how each token attends to other tokens in the sentence. Different layers and attention heads can be selected to explore the model's internal behavior.

![Attention Heatmap](Screenshot%202026-03-05%20212126.png)

---

### Token Importance Visualization

This visualization highlights which tokens contribute most strongly to the model's reasoning based on gradient analysis.

![Token Importance](Screenshot%202026-03-05%20212136.png)

---

## Installation

Clone the repository:

```
git clone https://github.com/swastik2475/llm-interpretability-toolkit.git
cd llm-interpretability-toolkit
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment.

Linux / macOS:

```
source venv/bin/activate
```

Windows:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Dashboard

Run the Streamlit application:

```
streamlit run dashboard/app.py
```

After running the command, open the local URL displayed in the terminal to access the dashboard.

---

## Running Tests

Automated tests can be executed with:

```
pytest
```

The repository also includes a CI workflow that runs tests automatically whenever code is pushed to the repository.

---

Future Improvements

Some potential extensions for this project include:

* Support for additional language models
* Interactive token highlighting in the input text
* Comparative analysis across multiple models
* Improved visualization of attention distributions

---

Author

Swastik Sharma
GitHub: https://github.com/swastik2475
