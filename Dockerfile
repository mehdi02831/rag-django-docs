FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data/index ./data/index
EXPOSE 8000
CMD ["uvicorn", "api:app", "--app-dir", "src/rag", "--host", "0.0.0.0", "--port", "8000"]