# mlops_linear_regression_major
Major Linear regression

This project implements a **complete MLOps pipeline** for a **Linear Regression model** using the California Housing dataset.  
The pipeline includes:  
Model training  
Testing with pytest  
Manual quantization  
Dockerization  
GitHub Actions CI/CD

## Repository structure

mlops_linear_regression_major
│
├── src/ # Source code
│ ├── train.py 
│ ├── predict.py 
│ ├── quantize.py 
│ ├── utils.py 
│ └── init.py
│
├── tests/ 
│ └── test_train.py
│
├── .github/workflows/ 
│ └── ci.yml
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore

## Repo setup
git clone https://github.com/g24ai1090/mlops_linear_regression_major.git 
cd cd mlops-linear-regression_major

## Create and activate conda env
conda create --name mlops_linear_regression_major python=3.10 -y
conda activate mlops_linear_regression_major

## Intsall dependencies
pip install -r requirements.txt

## Test the pipeline locally
python src/train.py
Result comes like 
----------------------------
Model trained successfully
 R2 Score: 0.5758
 Mean Squared Error: 0.5559
-----------------------------
python src/quantize.py
Result:
----------------------------
Model parameters quantized successfully (8-bit per-coefficient & 16-bit global)

 Full Model Comparison Table
| Version              | R² Score | MSE       | MAE       | Sample Predictions |
|----------------------|----------|-----------|-----------|-------------------|
| Sklearn (Unquantized) | 0.5758   | 0.5559 | 0.5332 | [0.72 1.76 2.71 2.84 2.6 ] |
| 8-bit Quantized      | -1163.9826   | 1526.6037 | 39.0550 | [-37. -37. -37. -37. -37.] |
| 8-bit Dequantized    | 0.5755   | 0.5563 | 0.5374 | [0.74 1.79 2.73 2.86 2.63] |
| 16-bit Quantized     | -1412345340153625.2500   | 1850750014114009.2500 | 34330109.7525 | [33443734.05 37641244.48 32208848.45 40777644.94 26061583.97] |
| 16-bit Dequantized   | 0.5756   | 0.5561 | 0.5340 | [0.72 1.77 2.71 2.84 2.61] |
-------------------------------------------

python src/predict.py
Result:
------------------------------
Predictions on test data:
Sample 1: 0.7191
Sample 2: 1.7640
Sample 3: 2.7097
Sample 4: 2.8389
Sample 5: 2.6047
-------------------------------

Run tests
python -m pytest
Result:
-------------------------------
=============== 3 passed in 2.68s ===============
------------------------------

Build docker image locally
1. docker build -t mlops-linear .
2. docker run mlops-linear


CI/CD Workflow
setup docker_username and docker_password keys going inside settings->actions->repository keys
1. git add --all
2. git commit -m "mlops pipeline on linear regression"
3. git push origin main

This would trigger action workflow in github

Model parameters quantized successfully (8-bit per-coefficient & 16-bit global)

 Full Model Comparison Table
| Version              | R² Score | MSE       | MAE       | Sample Predictions |
|----------------------|----------|-----------|-----------|-------------------|
| Sklearn (Unquantized) | 0.5758   | 0.5559 | 0.5332 | [0.72 1.76 2.71 2.84 2.6 ] |
| 8-bit Quantized      | -1163.9826   | 1526.6037 | 39.0550 | [-37. -37. -37. -37. -37.] |
| 8-bit Dequantized    | 0.5755   | 0.5563 | 0.5374 | [0.74 1.79 2.73 2.86 2.63] |
| 16-bit Quantized     | -1412345340153625.2500   | 1850750014114009.2500 | 34330109.7525 | [33443734.05 37641244.48 32208848.45 40777644.94 26061583.97] |
| 16-bit Dequantized   | 0.5756   | 0.5561 | 0.5340 | [0.72 1.77 2.71 2.84 2.61] |
