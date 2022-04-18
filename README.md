# GPT-3 Abstractive Summarization Project

## UI for summarization application

1. In root directory, run `src/venv/bin/activate` to activate virtualenv
2. Install dependencies with `pip install -r requirements.txt`
3. Start project ui with `streamlit run app.py`
4. Go to `localhost:8501`

## Project setup
1. Clone the repository with `git clone <repo_name>`
2. Create a python virtual environment with command `python -m venv <venv_name>`
3. Install dependencies with command `pip install -r requirements.txt`
4. Add a ".env" to project root with the following variables:
    - OPENAI_API_KEY: your personal API key from OpenAI
    - ENGINE: GPT-3 engine used (e.g. text-curie-001)
    - ORGANIZATION: the ID of the organization you registered in OpenAI
5. Go to src directory and start a production server with command `uvicorn main:app --reload`. The server will be hosted on localhost:8000.

## TODO

### Pipeline for training

#### Prepare data
    - Download datasets     DONE
    - Choose how many samples to draw from each dataset     DONE
    - Script to automate buildling a dataset, params

        `n`=how many samples

        `distribution`=propability distrib on dataset indices

        i.e. 500 samples from 4 datasets:

        `n=100`

        `distribution={0: 0.2, 1, 0.4, 2: 0.2, 3: 0.2}`
    
    - Create JSONL-file and use prep using `fine_tunes.prepare_data`

#### Build pipeline for testing hyperparams
