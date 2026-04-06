FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 7860
CMD ["uvicorn", "news_stock_env.space_app:app", "--host", "0.0.0.0", "--port", "7860"]
