# Metropolis-Hastings with Scalable Subsampling
Python scripts that can be used to reproduce the figures and tables presented in the paper "Metropolis-Hastings with Scalable Subsampling" (arxiv, 2024).

Some datasets used in the real-world applications can be found in the UCI machine learning repository, and the links are provided in the scripts. The road casualties dataset can be downloaded from the R package "stats19" as shown in the corresponding script. Finally, the US census dataset of the is provided in this repository.

We implemented all algorithms using two strategies: loop and vectorised operations. All algorithms in the main paper used for-loop operations to evaluate the likelihood terms. This strategy is generally slower, but it does make all algorithms comparable. In the Appendices, we show a comparison between loop and vectorised operations in the context of our simulations. As stated in the paper, we decided to adopt loop operations for the sake of having a fair comparison in the run-times amongst the considered algorithms due to the way the Scalable Metropolis-Hastings (Cornith et. al., 2019), which is implemented in C++, is designed.
