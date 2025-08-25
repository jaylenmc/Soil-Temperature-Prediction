# 🌡️ Soil Temperature Prediction
This project was part of my college summer research, where I explored machine learning approaches to predict soil temperature at different depths (ST_10, ST_50, ST_100) using weather 
and snow depth data from the Grand Forks, ND area. Although my career focus is software development, this project gave me hands-on experience with data preprocessing, model training, 
and evaluation, and helped me confirm that my main passion lies in building software systems.

# 🔍 Models Explored
### Fuzzy ANFIS (Adaptive Neuro-Fuzzy Inference System)
-	Tested membership functions (Trapezoidal, Triangular, Gaussian): Triangular performed the best 
-	Tested multiple optimizers (from Keras): Adamax performed the best
-	Used 100 epochs, batch size = 8
### Random Forest Regressor
-	Best results with base hyperparameters
### Linear Regression
-	Used base hyperparameters

# ⚙️ How To Run
1.	Clone this repo
git clone https://github.com/yourusername/soil-temp-prediction.git
cd soil-temp-prediction

2.	Install dependencies
pip install -r requirements.txt

3.	Run the script
python3 snow1.py

# 📝 Results
The models were evaluated using Root Mean Square Error (RMSE).
### ST_10
- Fuzzy ANFIS -> 1.422
- Random Forest Regressor -> 1.303
- Linear Regression -> 2.225
### ST_50
- Fuzzy ANFIS -> 1.919
- Random Forest Regressor -> 1.577
- Linear Regression -> 2.365
### ST_100
- Fuzzy ANFIS -> 1.972
- Random Forest Regressor -> 1.559
- Linear Regression -> 2.335

Across all tests, the Random Forest Regressor consistently achieved the lowest RMSE values, making it the best-performing model. The Fuzzy ANFIS model came in second. Random Forest won
because its flexible and handles complex data patterns without heavy tuning. Fuzzy ANFIS did reasonably well, but it’s harder to tune and needs more careful 
optimization to beat ensemble methods like Random Forest.
