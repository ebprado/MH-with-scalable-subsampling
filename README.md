# PyMHSS: A Python package that implements Metropolis-Hastings with Scalable Sampling algorithms

## Introduction
The `PyMHSS` package provides a set of Python scripts to reproduce the results of simulation experiments and real-world applications presented in [Prado, E.B., Nemeth, C. & Sherlock, C. Metropolis-Hastings with Scalable Subsampling. arxiv 
(2024)](https://arxiv.org/pdf/2407.19602). The package is designed to facilitate reproducibility.

## Installation

To install PyMHSS, use `pip` or `python3 -m pip`:

* Option 1: `pip install -i PyMHSS`
* Option 2: `python3 -m pip install -i PyMHSS`

## Repository folder structure

## Simulation experiments

* `01_Figure_1.py`: acceptance rates and ESS per second for SMH-1, Tuna and RWM. The results are based on synthetic datasets generated from a logistic regression model with $n = 31,622$ observations.

* `02_Figure_2.py`: acceptance rates of the proposed method with first-order control variates. For each combination of $\gamma$, $n$ and $d$, $10$ synthetic datasets are generated from a logistic regression model.

* `03_aux_generate_results.py`: generates the simulation results for Figures 3 and 4.

* `03_Figures_3_and_4.py`: optimal scaling for the MH-SS algorithms and acceptance rates, where $\alpha = \alpha_1 \times \alpha_2$, based on a logistic regression target in dimension $d=100$ with $n=30,000$ observations, and with covariates and true coefficients simulated as described in Section 5 of the paper. The efficiency metric (MSJD/E(B)) is plotted as a function of the scaling parameter ($\lambda$) and the empirical acceptance rate. 

* `04_aux_generate_results.py`: generates the simulation results for Figures 5, 6, 7, 8 and 9.

* `04_Figures_5_6_7_8_9.py`: ESS per second, ESS divided by E(B), average batch size for MH-SS, SMH and RWM for the logistic regression model. For RWM, the average batch size is $n$. In all figures, both axes are presented in the logarithm base 10.

* `05_Table_1.py`: acceptance rate, average batch size, ESS per second and ESS/E(B) for the Poisson regression model applied to synthetic data in $d=30$.
  
* `06_Figure_10.py`: Acceptance rates and mean squared jumping distance (MSJD) over E(B) based on a simulation experiment with a logistic regression model with dimension $d = 100$ and $n=100,000$.

## Real-world applications

Ideally, we would have liked to have all datasets analysed in the paper in this repository. However, some of them are too big to be stored here. The US census dataset is the only one provided in this repository. The datasets detection of gas mixtures and high-energy particle physics can be found in the [UCI machine learning repository](https://archive.ics.uci.edu/); see the links below.

### U.S. population survey 
* `usa_pop_survey_data.zip`: The 2018 United States Current Population Survey is a monthly household survey carried out by the U.S. Census Bureau and the U.S. Bureau of Labor Statistics that gathers information on the labour force for the population of the U.S. The data contain variables such as income, education, occupation, participation in welfare programs and health insurance.

* `usa_pop_survey.py`: We used a sample of $n = 500,000$ survey participants to model whether or not an individual has a total pre-tax personal income above $25,000$ U.S. dollars for the previous year, based on $10$ predictors.
  
### Detection gas
* `gas_sensor.py`: For a sample of size $n = 250,000$, we predict whether the concentration of Ethylene in the air, measured in parts per million, is above zero. Though we consider $d = 7$ continuous predictors only (including an intercept), each of which corresponds to a different model sensor, some of them are highly correlated.
  
* `data`: The dataset used in the analysis can be downloaded in this link: [detection of gas mixtures](https://archive.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures).

### Hepmass

* `hepmass.py`: For a sample of $n = 1, 000, 000$ observations from the training set, we analyse whether a new particle of unknown mass is observed. The dataset has originally $26$ continuous predictors and is split into a training set of $7$ million observations and a test set of $3.5$ million. 

* `data`: The dataset contains information about signatures of exotic particles obtained from a high-energy physics experiment. The binary response variable indicates whether a new particle of unknown mass is observed. The dataset can be found in this link: [high-energy particle physics](https://archive.ics.uci.edu/dataset/347/hepmass).

### Road casualties UK
* `road_casualties_UK.py`: We analyse the consolidated UK road casualties data from $2020$ to $2022$. We aim to model at the individual level of $n = 298,290$ accidents the number of casualties based on 8 predictors, of which 2 are continuous. In total, the linear predictor in our Poisson model with mean $\log(1 + e^{\eta})$ has $d=28$ parameters (including the intercept).

* `data`: The UK Department for Transport publishes annual road safety statistics as part of the Statistics and Registration Service Act 2007. The data include the accident's geographical coordinates, severity, speed limit of the road where the accident took place, details about the vehicles involved, weather conditions, road conditions, as well as time and date. The road casualties dataset can be downloaded from the R package `stats19` as

    ```r
    install.packages('stats19')
    library(stats19)
    
    if(curl::has_internet()) {
      
    dl_stats19(year = 2020, type = "casualty")
    dl_stats19(year = 2021, type = "casualty")
    dl_stats19(year = 2021, type = "casualty")
      
    casualties_2020 = read_casualties(year = 2020)
    casualties_2021 = read_casualties(year = 2021)
    casualties_2022 = read_casualties(year = 2022)
    
    }
    ```


## mhssteste
   * Python Package containing the implementation of the Metropolis-Hastings with Scalable Subsampling (MH-SS), random-walk Metropolis-Hastings (RWM), [Scalable Metropolis-Hastings (SMH)](http://proceedings.mlr.press/v97/cornish19a/cornish19a.pdf) and [TunaMH](https://proceedings.neurips.cc/paper/2020/file/e2a7555f7cabd6e31aef45cb8cda4999-Paper.pdf).

  * We implemented all algorithms using two strategies: `loop` and `vectorised` operations. All results in the main paper used for-loop operations to evaluate the likelihood terms. This strategy is generally slower, but it does make all algorithms comparable. In the Appendices, we show a comparison between `loop` and `vectorised` operations in the context of our simulations. As stated in the paper, we decided to adopt `loop` operations for the sake of having a fair comparison in the run-times amongst the considered algorithms due to the way the SMH algorithm, which is implemented in C++, is designed.
