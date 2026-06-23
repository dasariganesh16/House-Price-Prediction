# House Price Prediction

A simple Python project to predict house prices using linear regression on housing data.

## Contents

- `app.py` - main script that loads the dataset, trains a model, and predicts prices from user input
- `data/Housing.csv` - dataset used for training
- `.gitignore` - ignores virtual environment and Python build files

## Requirements

- Python 3.8+
- `pandas`
- `numpy`
- `scikit-learn`

## Setup

1. Create a Python virtual environment:

```powershell
python -m venv .venv
```

2. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install requirements:

```powershell
pip install pandas numpy scikit-learn
```

## Run

```powershell
python app.py
```

The script will train a linear regression model and then prompt for house features to predict the price.

## Notes

- The dataset uses encoded categorical values.
- This project is a starting point; consider improving the model, adding validation, and saving a trained model file.
