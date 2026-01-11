# Build Stage
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime Stage
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder to keep image small
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy Project Code
COPY . .

# Install the package itself in editable mode or just add to path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command: Run the Dashboard
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
