# Movie Recommendation System using Restricted Boltzmann Machine

## Project Overview

This project implements a movie recommendation system using a Restricted Boltzmann Machine (RBM). The system is trained on user movie ratings to predict user preferences and recommend movies.

## Features

- Utilizes the MovieLens 1M dataset
- Implements a Restricted Boltzmann Machine from scratch using PyTorch
- Trains the model on user ratings
- Converts ratings to binary (liked/not liked) for simplification
- Evaluates the model's performance on a test set

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/rbm-movie-recommender.git
   cd rbm-movie-recommender
   ```

2. Install the required packages:
   ```bash
   pip install torch numpy pandas
   ```

3. Download the MovieLens 1M dataset and place it in the project directory.

## Usage

Run the main script:
