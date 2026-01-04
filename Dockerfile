# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set timezone to UTC
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set work directory
WORKDIR /app

# Install system dependencies (if any needed for numpy/pandas compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose ports (Streamlit uses 8501)
EXPOSE 8501

# Define environment variable for unbuffered logging
ENV PYTHONUNBUFFERED=1

# Command to run bot AND dashboard (using a shell script or supervisor would be better, but simple approaches first)
# For now, we will just run the bot. The dashboard can be run in a separate container or background.
# Let's create a startup script.
RUN echo "#!/bin/bash\nstreamlit run dashboard.py & python trading_bot.py" > start.sh
RUN chmod +x start.sh

CMD ["./start.sh"]
