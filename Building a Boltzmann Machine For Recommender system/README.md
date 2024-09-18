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
## Code Structure

- `my_boltzmann_machine.py`: Main script containing data loading, preprocessing, RBM implementation, and training/testing loops.

Key components:
- Data loading and preprocessing
- RBM class definition
- Training loop
- Testing loop

## Data

The project uses the MovieLens 1M dataset, which should be organized as follows:
- `ml-1m/movies.dat`
- `ml-1m/users.dat`
- `ml-1m/ratings.dat`
- `ml-100k/u1.base` (training set)
- `ml-100k/u1.test` (test set)

## Model Details

The RBM is implemented with the following specifications:
- 100 hidden units
- Binary visible units (liked/not liked)
- Trained using contrastive divergence
- 10 epochs of training
- Batch size of 100

## Results

The script outputs:
- Training loss for each epoch
- Final test loss

## Future Improvements

- Implement hyperparameter tuning
- Add cross-validation
- Extend to deep belief networks
- Improve data preprocessing and feature engineering
  ## License

[MIT License](https://opensource.org/licenses/MIT)

## Contact

For any queries, please open an issue in this repository.
