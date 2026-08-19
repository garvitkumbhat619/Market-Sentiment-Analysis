FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK resources
RUN python -m nltk.downloader -d /usr/local/nltk_data stopwords punkt punkt_tab

ENV NLTK_DATA=/usr/local/nltk_data

COPY app.py .
COPY predict.py .
COPY models ./models

EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
