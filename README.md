# Prosperity Backtester & Analysis Tool

This tool allows backtesting trading algorithms for the Prosperity competition and analyzing results, including permutation testing.

Few notes about using the backtester that I think are important.
## Note
Your Trader.py need to be slightly modified for backtester to work. "import datamodel" needs to be changed to "import backtester.datamodel".

## Simulating Bot Reactions

How it works: This backtester uses a second orderbook that simulates bot reactions to orders unfilled by the first orderbook given to us. This is done through taking logs that we get from prosperity and attaching them to the backtester to create an empirical distribution of what can happen at each timestamp. Get the most accurate results, I tried to attach logs that vary orders as much as possible, having as much mutual information with each other and the true distibution - whilest having large enough size to capture all the liquidity data. I usually attach logs for vwap_d{i}, which buys and sells at vwap+-i, while neutralizing position at each timestep.

1. When setting bot behavior, I think there are only two that are useful given the structure of how I coded the liquidity orderbook. Setting it to "none" gets rid of the second orderbook. Setting it to "leq" makes it so that orders you submit can only be matched to the corresponding same price in the liquidity orderbook opr better. I think this is the most accurate.



## Setup

1.  **Prerequisites:**
    *   Python 3.9 or higher installed and added to your system's PATH.
    *   `pip` (Python package installer) available.

2.  **Clone or Download:**
    *   Get the project code: `git clone <repository_url>` or download the ZIP file.
    *   Navigate into the project directory in your terminal: `cd prosperity-backtester` (or your project's folder name).

3.  **Create Virtual Environment (Recommended):**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Linux / macOS
    python3 -m venv venv
    source venv/bin/activate
    ```
    *You should see `(venv)` at the beginning of your terminal prompt.*

4.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you have the `requirements.txt` file from the project).*

## Running the App

1.  **Make sure your virtual environment is active** (if you created one).
2.  **Navigate to the project directory** in your terminal (the one containing `app.py`).
3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
4.  The application should automatically open in your default web browser.

## Project Structure

*   `app.py`: The main Streamlit application file.
*   `requirements.txt`: Lists necessary Python packages.
*   `backtester/`: Contains the core backtesting logic (`backtester.py`, `util.py`, `constants.py`, `datamodel.py`, `permutation_utils.py`).
    *   `backtester/traders/`: Place your trader algorithm Python files here.
*   `data/`: Place your competition log files (`.log`) here.
*   `output/`: (Optional) Directory where output logs might be saved (if implemented).
*   `README.md`: This file.
*   `run.bat` / `run.sh`: (Optional) Helper scripts to automate setup and launch.



## Setup (Using Anaconda/Miniconda)

1.  **Prerequisites:**
    *   Anaconda or Miniconda installed.

2.  **Clone or Download:**
    *   Get the project code: `git clone <repository_url>` or download the ZIP file.
    *   Open an **Anaconda Prompt** or your standard terminal where `conda` is activated.
    *   Navigate into the project directory: `cd path/to/prosperity-backtester`

3.  **Create Conda Environment:**
    *   Run the following command in the project directory (where `environment.yml` is located):
      ```bash
      conda env create -f environment.yml
      ```
    *   This will create a new Conda environment named `prosperity_backtester_env` (or whatever you named it in the file) with all the specified packages. This might take a few minutes.

4.  **Activate the Environment:**
    *   Each time you want to run the app, you need to activate the environment:
      ```bash
      conda activate prosperity_backtester_env
      ```
    *   You should see `(prosperity_backtester_env)` at the beginning of your terminal prompt.

## Running the App

1.  **Make sure the Conda environment is active:** `conda activate prosperity_backtester_env`
2.  **Navigate to the project directory** in your terminal (the one containing `app.py`).
3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
4.  The application should automatically open in your default web browser.

## Updating the Environment

If the requirements change (`environment.yml` is updated), you can update your existing environment:
```bash
conda env update --name prosperity_backtester_env --file environment.yml --prune