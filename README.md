# Engineering Input Bot

This is a Streamlit-based chatbot application powered by Langchain and OpenAI's GPT models to assist users with input values required for a gas turbine simulation using the `tespy` library.

## Features

- Interactive chat to collect technical input for gas turbine simulations.
- Automated plotting of simulation results using `matplotlib`.
- Built with Streamlit for the user interface and Langchain for the conversational logic.

## Setup

### Prerequisites

- Docker installed on your machine.
- OpenAI API key (you can get it from the OpenAI platform).

### Running the Application

1. Clone this repository:
   ```bash
   git clone <repo_url>
   cd <repo_folder>
   docker build -t testpy .  
   docker run -p 8501:8501 testpy

