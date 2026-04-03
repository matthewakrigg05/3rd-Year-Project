This is the repository containing my university final year project. 

The aim of this project is to perform a comparative, and evaluative study on current sentiment analysis methods. This involves
comparing four tools: TextBlob, VADER, BERT and GPT-2. The study will weigh up the accuracy benefits against the computational
costs of these models.

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd 3rd-Year-Project
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env` and fill in your values
   - Required: `API_KEY=your_twitter_api_key_here`
   - Optional: `PYTHONDONTWRITEBYTECODE=1` (prevents __pycache__ creation - i find this annoying)

4. Run the project:
   - To collect data: `python src/collect_data.py` or `run_python.bat python src/collect_data.py`
   - To clean data: `python src/clean_data_set.py` or `run_python.bat python src/clean_data_set.py`
   - Run tests: `python -m unittest discover TESTING` or `run_python.bat python -m unittest discover TESTING`

   Note: Use `run_python.bat` instead of `python` to prevent __pycache__ creation, or set `PYTHONDONTWRITEBYTECODE=1` in your environment.

Project Structure Overview:
 - TESTING: The unit tests before development
 - src: What is developed and used as part of the projects outputs