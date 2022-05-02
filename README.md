# GPT-3 Abstractive Summarization Project

## documentation
Official OpenAI documentation: https://beta.openai.com/docs/introduction

## Project setup
1. Clone the repository with `git clone <repo_name>`
2. Add a ".env" to project root with the following variables:
    - OPENAI_API_KEY: your personal API key from OpenAI
    - ENGINE: GPT-3 engine used (e.g. text-curie-001)
    - ORGANIZATION: the ID of the organization you registered in OpenAI

## How to run the app
You only need to have Docker and docker-compose installed. You can install them from here: https://docs.docker.com/get-docker/

Then in the root of the project, run the following command in terminal: `docker-compose up`

And you're good to go! Go to `localhost:8501` in your browser and start using the summarization tool.