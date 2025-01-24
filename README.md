# Bati Bank Credit Risk Scoring System

This repository contains the implementation of a Credit Risk Scoring System for Bati Bank. The goal of this project is to develop machine learning models to assess credit risk and provide actionable insights for loan decisions.

## Project Structure
- **api/**: Contains the code for the REST API to serve ML models.
  - `app.py`: Main Flask/FastAPI application.
  - `test_api.py`: Unit tests for the API.
  - `utils.py`: Utility functions shared across API endpoints.
  - `models/`: Stores trained model files.

- **data/**: Placeholder for raw and processed data files.

- **notebooks/**: Jupyter notebooks for EDA, preprocessing, modeling, and results visualization.
  - `01_EDA.ipynb`: Exploratory Data Analysis.
  - `02_Preprocessing.ipynb`: Feature engineering and data preprocessing.
  - `03_Modeling.ipynb`: Model training and evaluation.
  - `04_Results.ipynb`: Results analysis and visualization.

- **reports/**: Contains interim and final project reports.

- **scripts/**: Modularized Python scripts for data processing and modeling.
  - `eda.py`: Functions for data exploration.
  - `preprocess.py`: Data preprocessing pipelines.
  - `modeling.py`: Model training and evaluation.
  - `utils.py`: Shared utility functions.

- **tests/**: Unit tests for scripts and API.

- **requirements.txt**: Python dependencies.

- **.gitignore**: Specifies files to ignore in version control.

## Installation
```bash
# Clone the repository
git clone https://github.com/Serkalem-negusse1/Bati_Bank_Credit_Risk.git
cd Bati_Bank_Credit_Risk

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Running the API
```bash
# Start the API server
python api/app.py
```

### Running Notebooks
Open the Jupyter notebooks in the `notebooks/` folder for exploratory and modeling tasks.

### Running Unit Tests
```bash
pytest tests/
```

## Features
- Data exploration and preprocessing pipelines.
- Machine learning models for credit risk scoring.
- REST API for serving predictions.
- Visualizations for insights and results.

## Deliverables
- Trained machine learning models.
- REST API for real-time predictions.
- Detailed reports and visualizations.


