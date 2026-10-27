# LSTM Next Word Predictor

A Streamlit application that uses a trained LSTM language model to predict the most likely next words for a phrase entered by the user.

The application loads a Keras model, tokenizer, and sequence-length metadata from local files, converts the input phrase into token IDs, and displays ranked predictions with probabilities.

## Features

- Interactive Streamlit user interface
- LSTM-based next-word prediction
- Top 1–10 prediction choices
- Probability ranking and visualization
- Automatic loading of model artifacts from the application directory
- Input validation for unknown words and invalid model output
- Cached model loading for faster repeated predictions

## Project Structure

```text
lstm-next-word-predictor/
├── app.py
├── lstm_model.h5
├── tokenizer.pkl
├── max_len.pkl
├── .gitignore
└── README.md
```

### Model Artifacts

| File | Purpose |
|---|---|
| `lstm_model.h5` | Trained Keras LSTM model |
| `tokenizer.pkl` | Fitted text tokenizer and vocabulary |
| `max_len.pkl` | Saved sequence-length metadata |

These files must remain compatible with one another. The tokenizer and sequence length should correspond to the model used for inference.

## Requirements

- Python 3.10 or newer
- TensorFlow
- Streamlit
- NumPy
- Pandas

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shubham190306/lstm-next-word-predictor.git
   cd lstm-next-word-predictor
   ```

2. Create and activate a virtual environment:

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install tensorflow streamlit numpy pandas
   ```

## Running the Application

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit will display a local URL, normally:

```text
http://localhost:8501
```

Open that URL in a browser, enter a phrase, choose the number of suggestions, and click **Predict next word**.

## How It Works

1. The application loads the tokenizer, maximum sequence length, and trained LSTM model.
2. The entered phrase is converted into integer token IDs.
3. The most recent tokens are padded or truncated to the model's expected input width.
4. The model generates scores for the vocabulary.
5. Scores are normalized into probabilities when necessary.
6. The application removes invalid or padding-token candidates and displays the highest-ranked words.

## Troubleshooting

### `ModuleNotFoundError` or unresolved TensorFlow imports

Make sure the virtual environment is activated and install the dependencies using the same Python interpreter that starts Streamlit:

```bash
python -m pip install tensorflow streamlit numpy pandas
python -m streamlit run app.py
```

In VS Code, select the same interpreter used by the virtual environment:

1. Open the Command Palette.
2. Select **Python: Select Interpreter**.
3. Choose `.venv`.

### Missing model files

The following files must be beside `app.py`:

```text
lstm_model.h5
tokenizer.pkl
max_len.pkl
```

### The phrase is not recognized

The tokenizer only recognizes words present in the training vocabulary. Try a phrase that uses words from the data used to train the model.

### Git error: `src refspec main does not match any`

Run Git commands from the repository directory and ensure that a commit exists:

```powershell
cd C:\next_word_prediction
git add .
git commit -m "Initial commit"
git branch -M main
git push -u origin main
```

## Security Note

The tokenizer is loaded from a pickle file. Pickle files can execute code during deserialization, so only use model artifacts from a trusted source.

## License

No license has been specified for this project yet. Add a license before distributing or reusing the code publicly.
