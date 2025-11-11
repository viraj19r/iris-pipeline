# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the app and model into the container
COPY ./app /app/app
COPY ./models /app/models

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.main when the container launches
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]