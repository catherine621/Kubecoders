# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy application files
COPY app.py /app/
COPY ./model /app/model

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
