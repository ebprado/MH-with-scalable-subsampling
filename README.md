# Metropolis-Hastings with Scalable Subsampling
## Repository Overview
This repository houses Python scripts that can be used to reproduce the results of simulation experiments and real-world applications presented in [Prado, E.B., Nemeth, C. & Sherlock, C. Metropolis-Hastings with Scalable Subsampling. arxiv (2024)](https://arxiv.org/pdf/2407.19602). In an attempt to facilitate reproducibility, we created a Python package and organised/named the scripts following the figures or tables their outputs are associated with. For any questions or issues, please refer to the contact information included in this repository.

Ideally, we would have liked to have all datasets analysed in the paper in this repository. However, some of them are too big to be stored here. The US census dataset is provided in this repository. The datasets [detection of gas mixtures](https://archive.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures) and [high-energy particle physics](https://archive.ics.uci.edu/dataset/347/hepmass) can be found in the [UCI machine learning repository](https://archive.ics.uci.edu/). Finally, the road casualties dataset can be downloaded from the R package `stats19` as

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

The package `mhteste` needs to be installed before running the scripts. This can downloaded from TestPyPI using the following command:

```python
pip install -i https://test.pypi.org/simple/ mhssteste
```

# Repository Folder Structure

### Simulation experiments:

* `01_Figure_1.py`: acceptance rates and ESS per second for SMH-1, Tuna and RWM. The results are based on synthetic datasets generated from a logistic regression model with $n = 10^{4.5} \approx 30,000$ observations.

* `02_Figure_2.py`: acceptance rates of the proposed method with first-order control variates. For each combination of $\gamma$, $n$ and $d$, $10$ synthetic datasets are generated from a logistic regression model.

* `03_aux_generate_results.py`: generates the simulation results for Figures 3 and 4.

* `03_Figures_3_and_4.py`: optimal scaling for the MH-SS algorithms and acceptance rates, where $\alpha = \alpha_1 \times \alpha_2$, based on a logistic regression target in dimension $d=100$ with $n=30,000$ observations, and with covariates and true coefficients simulated as described in Section 5. The efficiency metric (MSJD/E(B)) is plotted as function of the scaling parameter ($\lambda$) and the empirical acceptance rate. 

* `04_aux_generate_results.py`: generates the simulation results for Figures 5, 6, 7, 8 and 9.

* `04_Figures_5_6_7_8_9.py`: ESS per second, ESS divided by E(B), average batch size for MH-SS, SMH and RWM for the logistic regression model. For RWM, the average batch size is $n$. In all figures, both axes are presented in the logarithm base 10.

* 05_Table_1.py
  
* 06_Figure_10.py

### mhssteste:
   * Python Package containing the functions used to produce the forward modelling (Gaussian plume model) and inversion modelling (M-MALA-within-Gibbs) in the Chilbolton case study and the simulation study.
