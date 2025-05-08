# System imports
import os
import sys
import time

# Construct and print the path
functions_path = os.path.abspath(os.path.join(os.getcwd(), 'functions'))

# Add to sys.path if not already there
if functions_path not in sys.path:
    sys.path.append(functions_path)

# Imports
import pandas as pd

# Own libraries
import prospect_optimizer as po
import visualisation as vis
import weight_analysis as wa
import evaluation as ev
import other_optimizations as oo



toc = time.time()

parent_dir = os.getcwd() # speciale_repo

### INPUT PARAMETERS

# Latest data: '2016-12-01'
# Date range for the analysis
start_date = '1997-01-01'
end_date = '2007-01-01'

# Static parameter
date_tag = f"{start_date}_{end_date}"
# 11 years of data
# min_obs = 120

# Changeable?
n_predictors_to_use = 9

# strategies = ["conservative","aggressive"]
# lambda_values = [1.5, 1.75, 1.99, 2.25, 2.5]
# gamma_values = [0.12, 0.2, 0.25, 0.35, 0.5]

#Indlæser returnsdata
#datapath = os.path.join(parent_dir+'/data/returns_data.csv')

bma_returns = po.rolling_bma_returns(parent_dir, n_predictors_to_use=2, start_date=start_date, end_date=end_date)
tic = time.time()
print(f"Elapsed time: {(tic-toc):.2f}")