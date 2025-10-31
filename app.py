name: Email Automation Scheduler

on:
  schedule:
    # Run every 30 minutes to reduce overlap and load time
    - cron: '*/5 * * * *'
  workflow_dispatch:

jobs:
  run-email-agent:
    runs-on: ubuntu-latest

    services:
      # Define the Ollama service
      ollama_service:
        image: ollama/ollama:latest
        ports:
          # Maps the container port 11434 to the runner's localhost port 11434
          - 11434:11434
        options: >-
          --name ollama_service
          --health-cmd="ollama -v"
          --health-interval=10s
          --health-timeout=5s
          --health-retries=5

    env:
      EMAIL_ADDRESS: ${{ secrets.EMAIL_ADDRESS }}
      EMAIL_PASSWORD: ${{ secrets.EMAIL_PASSWORD }}
      IMAP_SERVER: imap.gmail.com
      SMTP_SERVER: smtp.gmail.com
      # CRITICAL FIX: Use localhost for the Python script to connect to the service
      OLLAMA_URL: http://localhost:11434/api/generate
      # Using a verified model tag
      OLLAMA_MODEL: mistral:7b-instruct-v0.2-q4_0
      LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }} 
      PYTHONUNBUFFERED: 1

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Dependencies
        run: |
          pip install --upgrade pip
          pip install requests

      # Wait for the Ollama service to start and become healthy
      - name: Wait for Ollama Service
        id: wait_ollama
        run: |
          echo "Waiting up to 2 minutes for Ollama service health..."
          for i in {1..24}; do
            if docker inspect --format='{{json .State.Health.Status}}' ollama_service | grep -q healthy; then
              echo "Ollama service is healthy!"
              break
            else
              echo "Still waiting for Ollama... ($i/24)"
              sleep 5
            fi
            if [ $i -eq 24 ]; then
              echo "Ollama service failed to become healthy. Aborting."
              exit 1
            fi
          done

      # Pull the Mistral model
      - name: Pull Mistral Model
        id: pull_model
        run: |
          # Stability check: Sleep briefly to ensure the Ollama CLI is fully ready
          echo "Waiting 5 seconds to ensure Ollama is fully ready..."
          sleep 5
          
          echo "Pulling model ${{ env.OLLAMA_MODEL }}..."
          # Execute the pull command inside the service container
          docker exec ollama_service ollama pull ${{ env.OLLAMA_MODEL }} || {
            echo "Retrying model pull after 10 seconds..."
            sleep 10
            docker exec ollama_service ollama pull ${{ env.OLLAMA_MODEL }}
          }

      - name: Run Email Agent
        id: run_agent
        run: |
          echo "Starting Email Automation..."
          # The Python script will now correctly connect via localhost:11434
          python email_agent.py
