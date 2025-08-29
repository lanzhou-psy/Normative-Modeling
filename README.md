# Normative modelling 🧠🧠


<img width="501" alt="image" src="https://github.com/predictive-clinical-neuroscience/NM_educational_OHBM24/assets/23728822/c3e8a638-ae40-4f66-a8ec-a818c6660706">


Normative modelling has revolutionized our approach to identifying effects in neuroimaging, enabling a shift from group-level statistics to subject-level inferences. The use of extensive population data, now readily accessible, allows for a more comprehensive understanding of healthy brain development as well as our understanding of psychiatric and neurological conditions. Beyond individualized predictions, advantages of normative modelling include the transfer of predictions to unseen sites, data harmonization, and flexibility and adaptability to various brain imaging data types and distributions. However, the considerable power of these models requires careful handling and interpretation.
I suggest you could read these papers first:
1. https://www.nature.com/articles/s41380-019-0441-1
2. https://www.sciencedirect.com/science/article/pii/S0006322316000020

Here’s the idea step by step:

- Collect a big dataset – You scan the brains of many healthy people across different ages, sexes, etc.

- Build the model of “normal” – The model learns what brain measures (like thickness of the cortex, volume of regions, or connectivity) usually look like at different ages and conditions. It’s like building a growth chart, but for the brain.

- Compare individuals and compute the deviations – When you scan one person’s brain, you check where they fall compared to the “normal” curve. If their value is within the typical range, that’s expected. If it’s much higher or lower than expected, it shows an individual difference that might be linked to disease, resilience, or other traits.


```
# Make sure to click the restart runtime button at the
# bottom of this code blocks' output (after you run the cell)
! pip install pcntoolkit==0.30

! git clone https://github.com/CharFraza/CPC_ML_tutorial.git

# we need to be in the CPC_ML_tutorial folder when we import the libraries in the code block below,
# because there is a function called nm_utils that is in this folder that we need to import
import os
os.chdir('/content/CPC_ML_tutorial/')

```

Several packages should be imported first.
```
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import seaborn as sns

from pcntoolkit.normative import estimate, predict, evaluate
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix
from nm_utils import calibration_descriptives, remove_bad_subjects, load_2d
```

The data are assumed to be in CSV format and will be loaded as pandas dataframes
Generally the raw data will be in a different location to the analysis
The data can have arbitrary columns but some are required by the script, i.e. 'age', 'sex' and 'site', plus the phenotypes you wish to estimate (see below)
```
# where the raw data are stored
data_dir = '/data/'

# where the analysis takes place
root_dir = '/data analysis/'
out_dir = os.path.join(root_dir,'models','test')

# create the output directory if it does not already exist
os.makedirs(out_dir, exist_ok=True)
```

Now we load the data.
We will load one pandas dataframe for the training set and one dataframe for the test set. We also configure a list of site ids.
```
df_tr = pd.read_csv(os.path.join(data_dir,'train_data.csv'), index_col=0)
df_te = pd.read_csv(os.path.join(data_dir,'test_data.csv'), index_col=0)

# extract a list of unique site ids from the training set
site_ids =  sorted(set(df_tr['site'].to_list()))
```
## Configure model parameters
Now, we configure some parameters for the regression model we use to fit the normative model. Here we will use a 'warped' Bayesian linear regression model. To model non-Gaussianity, we select a sin arcsinh warp and to model non-linearity, we stick with the default value for the basis expansion (a cubic b-spline basis set with 5 knot points). Since we are sticking with the default value, we do not need to specify any parameters for this, but we do need to specify the limits. We choose to pad the input by a few years either side of the input range. We will also set a couple of options that control the estimation of the model.

```
# check the min & max age of the dataset, use this info to update the xmin & xmax variables in the code block below.
df_tr['age'].describe()
# which data columns do we wish to use as covariates?
# You could add additional covariates from your own dataset here that you wish to use as predictors.
# However, for this tutorial today we will keep it simple and just use age & sex.
# Maybe discuss with your partner ideas you have for other covariates you would like to include.
cols_cov = ['age','sex']

# which warping function to use? We can set this to None in order to fit a vanilla Gaussian noise model
warp =  'WarpSinArcsinh'
warp = None

# limits for cubic B-spline basis
# check the min & max ages of the dataframes, add 5 to the max
# and subtract 5 from the min and adjust these variables accordingly
xmin = 13 # set this variable
xmax = 92 # set this variable

# Do we want to force the model to be refit every time?
# When training normative model from scratch like we are doing in this notebook (not re-using a pre-trained model),
# this variable should be = True
force_refit = True

# Absolute Z treshold above which a sample is considered to be an outlier (without fitting any model)
outlier_thresh = 7
```
## fit the model
Now we fit the models. This involves looping over the IDPs we have selected. We will use a module from PCNtoolkit to set up the design matrices, containing the covariates, fixed effects for site and nonlinear basis expansion.
```
for idp_num, idp in enumerate(idp_ids):
    print('Running IDP', idp_num, idp, ':')

    # set output dir
    idp_dir = os.path.join(out_dir, idp)
    os.makedirs(os.path.join(idp_dir), exist_ok=True)
    os.chdir(idp_dir)

    # extract the response variables for training and test set
    y_tr = df_tr[idp].to_numpy()
    y_te = df_te[idp].to_numpy()

    # remove gross outliers and implausible values
    yz_tr = (y_tr - np.mean(y_tr)) / np.std(y_tr)
    yz_te = (y_te - np.mean(y_te)) / np.std(y_te)
    nz_tr = np.bitwise_and(np.abs(yz_tr) < outlier_thresh, y_tr > 0)
    nz_te = np.bitwise_and(np.abs(yz_te) < outlier_thresh, y_te > 0)
    y_tr = y_tr[nz_tr]
    y_te = y_te[nz_te]

    # write out the response variables for training and test
    resp_file_tr = os.path.join(idp_dir, 'resp_tr.txt')
    resp_file_te = os.path.join(idp_dir, 'resp_te.txt')
    np.savetxt(resp_file_tr, y_tr)
    np.savetxt(resp_file_te, y_te)

    # configure the design matrix
    X_tr = create_design_matrix(df_tr[cols_cov].loc[nz_tr],
                                site_ids = df_tr['site'].loc[nz_tr],
                                basis = 'bspline',
                                p = 3,
                                nknots = 5,
                                xmin = xmin,
                                xmax = xmax)
    X_te = create_design_matrix(df_te[cols_cov].loc[nz_te],
                                site_ids = df_te['site'].loc[nz_te],
                                all_sites=site_ids,
                                basis = 'bspline',
                                p = 3,
                                nknots = 5,
                                xmin = xmin,
                                xmax = xmax)

    # configure and save the covariates
    cov_file_tr = os.path.join(idp_dir, 'cov_bspline_tr.txt')
    cov_file_te = os.path.join(idp_dir, 'cov_bspline_te.txt')
    np.savetxt(cov_file_tr, X_tr)
    np.savetxt(cov_file_te, X_te)

    if not force_refit and os.path.exists(os.path.join(idp_dir, 'Models', 'NM_0_0_estimate.pkl')):
        print('Making predictions using a pre-existing model...')
        suffix = 'predict'

        # Make prdictsion with test data
        predict(cov_file_te,
                alg='blr',
                respfile=resp_file_te,
                model_path=os.path.join(idp_dir,'Models'),
                outputsuffix=suffix)
    else:
        print('Estimating the normative model...')
        estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te,
                 testcov=cov_file_te, alg='blr', optimizer = 'l-bfgs-b',
                 savemodel=True, warp=warp, warp_reparam=True)
        suffix = 'estimate'

print(pd.DataFrame(X_te))
```

## Compute error metrics
In this section we compute the following error metrics for all IDPs (all evaluated on the test set): assess the goodness of fit between the predicted probabilities of a model and the actual observed outcomes.

Negative log likelihood (NLL): NLL assesses the goodness of fit between the predicted probabilities of a model and the actual observed outcomes. In this case, it measures the discrepancy between the predicted probabilities of the model for the IDPs (Independent Data Points) and the actual outcomes on the test set.
Explained variance (EV): EV assesses how much of the total variation in the dependent variable (IDP) is explained by the independent variables. In the context of this analysis, it quantifies the extent to which the independent variables account for the variability observed in the IDPs on the test set.
Mean standardized log loss (MSLL): MSLL takes into account both the mean error and the estimated prediction variance. It is used to evaluate the performance of the model, and in this case, a lower MSLL indicates a better-fitting model for the IDPs on the test set.
Bayesian information criteria (BIC): BIC is a model selection criterion that balances the goodness of fit to the data with the model's complexity. It penalizes models with higher flexibility and aims to find the best trade-off. Lower BIC scores indicate models that better explain the IDPs on the test set while considering the model complexity.
Skew and Kurtosis of the Z-distribution: Skewness and kurtosis are statistical measures used to assess the shape and characteristics of a distribution. They provide information about how well the warping function performed in terms of capturing the departure from a normal distribution for the IDPs.
```
# initialise dataframe we will use to store quantitative metrics
blr_metrics = pd.DataFrame(columns = ['eid', 'NLL', 'EV', 'MSLL', 'BIC','Skew','Kurtosis'])

for idp_num, idp in enumerate(idp_ids):
    idp_dir = os.path.join(out_dir, idp)

    # load the predictions and true data. We use a custom function that ensures 2d arrays
    # equivalent to: y = np.loadtxt(filename); y = y[:, np.newaxis]
    yhat_te = load_2d(os.path.join(idp_dir, 'yhat_' + suffix + '.txt'))
    s2_te = load_2d(os.path.join(idp_dir, 'ys2_' + suffix + '.txt'))
    y_te = load_2d(os.path.join(idp_dir, 'resp_te.txt'))

    with open(os.path.join(idp_dir,'Models', 'NM_0_0_estimate.pkl'), 'rb') as handle:
        nm = pickle.load(handle)

    # compute error metrics
    if warp is None:
        metrics = evaluate(y_te, yhat_te)

        # compute MSLL manually as a sanity check
        y_tr_mean = np.array( [[np.mean(y_tr)]] )
        y_tr_var = np.array( [[np.var(y_tr)]] )
        MSLL = compute_MSLL(y_te, yhat_te, s2_te, y_tr_mean, y_tr_var)
    else:
        warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1]
        W = nm.blr.warp

        # warp predictions
        med_te = W.warp_predictions(np.squeeze(yhat_te), np.squeeze(s2_te), warp_param)[0]
        med_te = med_te[:, np.newaxis]

        # evaluation metrics
        metrics = evaluate(y_te, med_te)

        # compute MSLL manually
        y_te_w = W.f(y_te, warp_param)
        y_tr_w = W.f(y_tr, warp_param)
        y_tr_mean = np.array( [[np.mean(y_tr_w)]] )
        y_tr_var = np.array( [[np.var(y_tr_w)]] )
        MSLL = compute_MSLL(y_te_w, yhat_te, s2_te, y_tr_mean, y_tr_var)

    Z = np.loadtxt(os.path.join(idp_dir, 'Z_' + suffix + '.txt'))
    [skew, sdskew, kurtosis, sdkurtosis, semean, sesd] = calibration_descriptives(Z)

    BIC = len(nm.blr.hyp) * np.log(y_tr.shape[0]) + 2 * nm.neg_log_lik

    blr_metrics.loc[len(blr_metrics)] = [idp, nm.neg_log_lik, metrics['EXPV'][0],
                                         MSLL[0], BIC, skew, kurtosis]

display(blr_metrics)

blr_metrics.to_csv(os.path.join(out_dir,'blr_metrics.csv'))
```



