# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run Streamlit when the container launches
CMD ["streamlit", "run", "--server.enableCORS", "false", "app.py"]
