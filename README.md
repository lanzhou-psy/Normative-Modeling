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

