# Metropolis-Hastings with Scalable Subsampling
## Repository Overview
This repository houses Python scripts that can be used to reproduce the results of simulation experiments and real-world applications presented in [Prado, E.B., Nemeth, C. & Sherlock, C. Metropolis-Hastings with Scalable Subsampling. arxiv (2024)](https://arxiv.org/pdf/2407.19602). In an attempt to make the reproducibility easier, we organise and name the scripts following the figures or tables their results are associated with. For any questions or issues, please refer to the contact information included in this repository.

Some datasets used in the real-world applications can be found in the [UCI machine learning repository](https://archive.ics.uci.edu/), and the links are provided in the scripts. The road casualties dataset can be downloaded from the R package "stats19" as shown in the corresponding script. Finally, the US census dataset is provided in this repository.

The package "mhteste" needs to be installed before running the scripts. This can downloaded from TestPyPI using the following command:

```python
pip install mhssteste
```

# Repository Folder Structure

### Code:
   * "Chilbolton_Case_Study" contains all the notebooks for the case study. Corresponds to Section 5 in the main paper and Supplementary Material B.2.

        * "Data Processing" shows how the Chilbolton dataset was cleaned and formatted for the case study. Corresponds to Supplementary Material B.2.1.

        - "Exploratory Data Analysis" determining the atmospheric stability class at time of the Chilbolton dataset measurements. Corresponds to Supplementary Material B.2.4.

        - "Inversion"  Manifold Metropolis-adjusted Langevin algorithm within Gibbs (M-MALA-within-Gibbs) parameter estimation for Source 1, Source 2, Source 3 and Source 4 from the Chilbolton dataset. Corresponds to Section 5 in the main paper and Supplementary Material B.2.5, B.2.6, and B.2.7.
    

   * "Simulation Study" contains all the notebooks for the simulation study. Corresponds to Section 4 in the main paper and Supplementary Material B.1.

        - "Simulation Study Code.ipynb" this notebook was used to produce all the results of Section 4 in the main paper. 

        - "Simulation Study Plots.ipynb" this notebook was used to produce all the plots of Section 4 in the main paper and in Supplementary Material B.1. 



### Data:
   * "Chilbolton_data_files" contains preprocessed .txt and postprocessed .pkl files of Chilbolton CH4 measurements and wind fields.

        - "Preprocessed" .txt files for each controlled release measurements and wind field.

        - "Postprocessed" .pkl files of (1) laser dispersion spectrometer measurements for Source 1, Source 2, Source 3, and Source 4 emissions, (2) reflector locations, (3) corresponding wind fields, and (4) source emission rates and locations.
    

   * "MCMC_chains" Chilbolton case study and simulation study samples of source emission rate, location, background gas concentration, measurement error variance and Gaussian plume dispersion parameters (for the non atmospheric stability class-based models).

        - "Chilbolton_case_study" sample chains for atmospheric stability class-based models (Smith and Briggs parametrizations) and non atmospheric stability class-based models (est. Draxler and est. Smith parametrizations) for Source 1, Source 2, Source 3, and Source 4 datasets.

        - "Simulation_study" sample chains for (1) varying parameters (a) WDC: wind direction coverage [degrees°], (b) DPV: dispersion parameter values, (c) SER: source emission rate [kg/s], (d) DTS: distance between the source and sensors [m], (e) OPS: number of observations per sensor, and (f) SL: sensor layout. And in-depth study of wind direction coverages and sensor layouts; (2) estimating dispersion parameters vs fixing them based on the atmospheric stability class.



### Paper:
   * Contains the manuscript for "Probabilistic Inversion Modelling of Gas Emissions: A Gradient-Based MCMC Estimation of Gaussian Plume Parameters", Supplementary Material A, and Supplementary Material B. Co-authors: Christopher Nemeth, Matthew Jones, Philip Jonathan.



### Sourceinversion:
   * Package containing the functions used to produce the forward modelling (Gaussian plume model) and inversion modelling (M-MALA-within-Gibbs) in the Chilbolton case study and the simulation study.
