# Use official Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Upgrade pip, setuptools, wheel first
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the entire project
COPY . .

# Expose port (change if your app uses different port)
EXPOSE 8050

# Command to run your Dash app
CMD ["python", "softsensing_app_v1.py"]
