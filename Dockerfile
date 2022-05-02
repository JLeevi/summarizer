FROM python:3.9

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

COPY . .

ENTRYPOINT [ "streamlit", "run" ]
CMD [ "src/app.py" ]