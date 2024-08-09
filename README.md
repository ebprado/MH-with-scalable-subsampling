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

* `01_Figure_1.py` generates acceptance rates and ESS per second for SMH-1, Tuna and RWM. The results are based on synthetic datasets generated from a logistic regression model with $n = 10^{4.5} \approx 30,000$ observations.
02_Figure_2.py
03_Figures_3_and_4.py
03_aux_generate_results.py
04_Figures_5_6_7_8_9.py
04_aux_generate_results.py
05_Table_1.py
06_Figure_10.py

   * "Chilbolton_Case_Study" contains all the notebooks for the case study. Corresponds to Section 5 in the main paper and Supplementary Material B.2.

        * "Data Processing" shows how the Chilbolton dataset was cleaned and formatted for the case study. Corresponds to Supplementary Material B.2.1.

        - "Exploratory Data Analysis" determining the atmospheric stability class at time of the Chilbolton dataset measurements. Corresponds to Supplementary Material B.2.4.

        - "Inversion"  Manifold Metropolis-adjusted Langevin algorithm within Gibbs (M-MALA-within-Gibbs) parameter estimation for Source 1, Source 2, Source 3 and Source 4 from the Chilbolton dataset. Corresponds to Section 5 in the main paper and Supplementary Material B.2.5, B.2.6, and B.2.7.
    

   * "Simulation Study" contains all the notebooks for the simulation study. Corresponds to Section 4 in the main paper and Supplementary Material B.1.

        - "Simulation Study Code.ipynb" this notebook was used to produce all the results of Section 4 in the main paper. 

        - "Simulation Study Plots.ipynb" this notebook was used to produce all the plots of Section 4 in the main paper and in Supplementary Material B.1. 

### mhssteste:
   * Python Package containing the functions used to produce the forward modelling (Gaussian plume model) and inversion modelling (M-MALA-within-Gibbs) in the Chilbolton case study and the simulation study.
