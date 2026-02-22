# Use a lightweight Python image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

# Run the application using Gunicorn (production-grade server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
