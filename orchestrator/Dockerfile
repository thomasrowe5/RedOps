FROM python:3.11-slim

RUN groupadd --system nonroot \
    && useradd --system --gid nonroot --create-home redops \
    && useradd --system --gid nonroot --create-home nonroot \
    && mkdir -p /app/data

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
RUN chown -R nonroot:nonroot /app

USER nonroot

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
