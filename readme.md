# **Optimal Sleep Pattern Recommendations using Machine Learning on Small Data**

The source code of the experiments from the corresponding paper.  
The used Python interpreter version is 3.7, but any version higher than 3.5 should be suitable.  
Before running, install libraries from `requirements.txt` using `pip`.

## Preparation 
The data was not pushed into this repository.
To get it, use the references from the paper.  
Put the original data into `data` folder. Rename Sleep Cycle csv files to `sd1.csv`, `sd2.csv`, and `sd3.csv`.
Rename the manually collected one to `simple1.csv`. Next, run `preprocess/preprocess_data.py`.  
After that, there should be files `sleepdata.csv`, `sleepdata_classified.csv`, and `sleepdata_classified_2.csv` in the 
`data` folder.

## Running

#### Drawing the Data Analysis Plot
To draw the plot with data analysis, run `preprocess/analyze.py`.

#### Running the Experiments
To run the actual experiments, run `main.py`.  
When script is done, there would be four Excel files with the results of the experiments.