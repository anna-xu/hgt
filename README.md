This repo contains data & analysis code for the latest version of the manuscript Xu et al., 2026, "Do you ever get tired of being wrong? The unique impact of feedback on subjective experiences of effort"

* Preprocessed data can be found in the folder `data`.
* The main script for analysis is `analysis.m`. 

#### Updates

09/17/2026:
- All analyses scripts, including the main script `analysis.m`, has been updated with the models ran in the manuscript. 
- Additional analyses to check the stability of feedback results when excluding extreme cases of feedback missingness are in `check_fb_exclusions.m`
- Preprocessed data (in the `data` folder) has been updated to be more streamlined for analyses (setup is now no longer required, and the data can be imported directly). Several variables have been updated (see wiki for more details on variable names).

12/19/2024: 
- `analysis.m` has been updated with models for additional analyses
- `plot_figures_s2_s3.py` includes code for plotting figures S2 and S3
    - `data/hgt_data.csv` contains a .csv version of `data/HandgripData.mat`
- Preprocessed data has been updated following a correction to an error in the previous release