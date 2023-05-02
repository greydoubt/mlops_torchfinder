# Use the official Python image as the base image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app/requirements.txt
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variable to the default path to the training data
ENV TRAINING_DATA /app/training_data

# Copy the server files into the container at /app
COPY server.py templates/ static/ deep_learning/ ./

# Expose the port that the server is listening on
EXPOSE 5000

# Start the server when the container starts
CMD ["python", "server.py"]
