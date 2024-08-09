# Metropolis-Hastings with Scalable Subsampling

This repository houses Python scripts that can be used to reproduce the results of simulation experiments and real-world applications presented in [Prado, E.B., Nemeth, C. & Sherlock, C. Metropolis-Hastings with Scalable Subsampling. arxiv (2024)](https://arxiv.org/pdf/2407.19602). In an attempt to facilitate reproducibility, we created a Python package (see intructions below) and organised/named the scripts following the figures or tables their outputs are associated with. For any questions or issues, please refer to the contact information included in this repository.

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

# Repository folder structure

### Simulation experiments:

* `01_Figure_1.py`: acceptance rates and ESS per second for SMH-1, Tuna and RWM. The results are based on synthetic datasets generated from a logistic regression model with $n = 10^{4.5} \approx 30,000$ observations.

* `02_Figure_2.py`: acceptance rates of the proposed method with first-order control variates. For each combination of $\gamma$, $n$ and $d$, $10$ synthetic datasets are generated from a logistic regression model.

* `03_aux_generate_results.py`: generates the simulation results for Figures 3 and 4.

* `03_Figures_3_and_4.py`: optimal scaling for the MH-SS algorithms and acceptance rates, where $\alpha = \alpha_1 \times \alpha_2$, based on a logistic regression target in dimension $d=100$ with $n=30,000$ observations, and with covariates and true coefficients simulated as described in Section 5. The efficiency metric (MSJD/E(B)) is plotted as function of the scaling parameter ($\lambda$) and the empirical acceptance rate. 

* `04_aux_generate_results.py`: generates the simulation results for Figures 5, 6, 7, 8 and 9.

* `04_Figures_5_6_7_8_9.py`: ESS per second, ESS divided by E(B), average batch size for MH-SS, SMH and RWM for the logistic regression model. For RWM, the average batch size is $n$. In all figures, both axes are presented in the logarithm base 10.

* `05_Table_1.py`: acceptance rate, average batch size, ESS per second and ESS/E(B)$ for the Poisson regression model applied to synthetic data in $d=30$.
  
* `06_Figure_10.py`: Acceptance rates and mean squared jumping distance (MSJD) over E(B) based on a simulation experiment with a logistic regression model with dimension $d = 100$ and $n=100,000$.

### mhssteste:
   * Python Package containing the implementation of the Metropolis-Hastings with Scalable Subsampling (MH-SS), random-walk Metropolis-Hastings (RWM), [Scalable Metropolis-Hastings (SMH)](http://proceedings.mlr.press/v97/cornish19a/cornish19a.pdf) and [TunaMH](https://proceedings.neurips.cc/paper/2020/file/e2a7555f7cabd6e31aef45cb8cda4999-Paper.pdf).

  * We implemented all algorithms using two strategies: `loop` and `vectorised` operations. All results in the main paper used for-loop operations to evaluate the likelihood terms. This strategy is generally slower, but it does make all algorithms comparable. In the Appendices, we show a comparison between `loop` and `vectorised` operations in the context of our simulations. As stated in the paper, we decided to adopt `loop` operations for the sake of having a fair comparison in the run-times amongst the considered algorithms due to the way the SMH algorithm, which is implemented in C++, is designed.
