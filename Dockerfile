FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY vecsearch ./vecsearch
RUN pip install --no-cache-dir .
RUN useradd --create-home appuser
USER appuser
CMD ["python", "-m", "vecsearch"]
