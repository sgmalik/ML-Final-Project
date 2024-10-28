# ML-Final-Project
**Cierra Church and Surya Malik**

# House Rent Estimation Machine Learning Project

This repository contains a machine learning project aimed at estimating the monthly rent of houses based on factors such as the number of bedrooms, location, and availability of utilities. The goal of this project is to develop a predictive model that can assist users and stakeholders in understanding and forecasting rental prices in various regions.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Modeling Approach](#modeling-approach)
- [Next Steps](#next-steps)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

The purpose of this project is to analyze various housing attributes and develop a machine learning model that can accurately predict the rental prices.

## Data

The dataset used in this project contains information on various housing attributes, such as:
- X1​: Price of the Houses (Integer)
- X2​: Area of the House (Integer)
- X3: Number of Bedrooms in the House (Integer)
- X4​: Number of Bathrooms in the House (Integer)
- X5​: Number of Stories in the house (Integer)
- X6​: Mainroad Connection (Boolean)
- X7: Guestroom (Boolean)
- X8: Basement (Boolean)
- X9: Hot Water Heating (Boolean)
- X10: Air Conditioning (Boolean)
- X11: Parking Spots (Integer)
- X12: Furnishing Status (String)


**Note:** The data used for this project may be specific to a certain geographic region and may not generalize well to other regions.

## Modeling Approach

The following steps were taken to develop the rental price prediction model:

1. **Data Exploration:** Understanding the structure of the data and identifying key features.
2. **Data Preprocessing:** Handling missing values, encoding categorical variables, and scaling numerical features.
3. **Feature Engineering:** Creating new features to improve the model's predictive capability.
4. **Model Training:** Several regression models were evaluated, including linear regression, decision trees, and ensemble methods.
5. **Model Evaluation:** Models were evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
  
## Next Steps

The project is ongoing, and the following steps are planned to improve the model and expand the analysis:
- **Enhancing Data Quality:** Collect more data to increase the diversity of locations and housing types.
- **Feature Expansion:** Consider incorporating additional features, such as neighborhood crime rates, proximity to public transport, and school ratings.
- **Model Optimization:** Experiment with hyperparameter tuning and more advanced models (e.g., Gradient Boosting Machines, Neural Networks).
- **Deployment:** Develop a web-based interface for users to input house attributes and receive rental price predictions in real-time.

## Getting Started

To get a copy of the project up and running on your local machine, follow the instructions below.

### Prerequisites

Make sure you have the following installed:
- Python 3.8+
- Jupyter Notebook or Jupyter Lab
- Git

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sgmalik/ML-Final-Project.git
   cd ML-Final-Project
   ```
2. **Create Virtual Environment**
    ```python
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3. **Open Jupyter Notebook and Run Cells**
    ```python
    jupyter notebook main.ipynb
    ```
**MAKE SURE DATASET IN PROPER DIRECTORY**

