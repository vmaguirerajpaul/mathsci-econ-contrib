![](acadmathsci-logo.png)
# mathsci-econ-contrib
Quantifying the UK economic contribution of the mathematical sciences in 2023 and beyond.


## Background
In a 2024 report, the Academy for the Mathematical Sciences (AcadMathSci) estimated that 4.2 million people were directly employed in mathematical science occupations, directly adding some £490 bn in gross value (20% of total GVA) to the UK's economy in 2023. 

This repository contains all the code used to perform the analysis leading to those estimates.

## How to use
The code has been tested with Python > 3.9.7 (though is likely to work with earlier versions of Python 3.x).
### Dependencies
| Library | Required? | Relevance |
| :-- | :-- | :-- |
numpy | ✅ | Performing array and matrix operations |
pandas | ✅ | Working with heterogeneous tabular data |
openpyxl | ✅ | Reading ONS datasets from .xlsx (Excel) files |
xlrd | ✅ | Reading ONS datasets from .xls (Excel) files |
jupyterlab | ❌ | Running code from Jupyter notebooks (optional) |

### Running code
Clone or download the repository:
```
git clone https://github.com/vmaguirerajpaul/mathsci-econ-contrib
```
The main analysis is run from ```01_main_analysis.ipynb``` while the salary-distribution analysis is run from ```02_salary_analysis.ipynb```.

If you'd rather not use Jupyter notebooks for the main analysis, navigate to the src directory
```
cd mathsci-econ-contrib/src/
```
then run the pre-processing scripts in this order:
```
python3 ./preprocessing/import_deloitte_data.py
python3 ./import_deloitte_data.py
python3 ./compute_ONS_SOC_relationships.py
```
Finally, run the main scripts:
```
python3 compute_all_SOC_maps.py
python3 estimate_MS_workforce.py
python3 estimate_GVA_from_workforce.py
```

## License
Distributed under the MIT License. See LICENSE.txt for more information.
## Contact
policyunit [at] AcadMathSci [dot] org [dot] uk
## Acknowledgements
AcadMathSci's Policy Unit is grateful to  to Prof. Peter Elias CBE, Dr Daniel Herring, Prof. Colin Turner, and the economics team at Public First for their independent review of the methodology and code underpinning this work. 
