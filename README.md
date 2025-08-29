# Normative modelling 🧠🧠


<img width="501" alt="image" src="https://github.com/predictive-clinical-neuroscience/NM_educational_OHBM24/assets/23728822/c3e8a638-ae40-4f66-a8ec-a818c6660706">


Normative modelling has revolutionized our approach to identifying effects in neuroimaging, enabling a shift from group-level statistics to subject-level inferences. The use of extensive population data, now readily accessible, allows for a more comprehensive understanding of healthy brain development as well as our understanding of psychiatric and neurological conditions. Beyond individualized predictions, advantages of normative modelling include the transfer of predictions to unseen sites, data harmonization, and flexibility and adaptability to various brain imaging data types and distributions. However, the considerable power of these models requires careful handling and interpretation.

Here’s the idea step by step:

- Collect a big dataset – You scan the brains of many healthy people across different ages, sexes, etc.

- Build the model of “normal” – The model learns what brain measures (like thickness of the cortex, volume of regions, or connectivity) usually look like at different ages and conditions. It’s like building a growth chart, but for the brain.

- Compare individuals and compute the deviations – When you scan one person’s brain, you check where they fall compared to the “normal” curve. If their value is within the typical range, that’s expected. If it’s much higher or lower than expected, it shows an individual difference that might be linked to disease, resilience, or other traits.


```
# Make sure to click the restart runtime button at the
# bottom of this code blocks' output (after you run the cell)
! pip install pcntoolkit==0.30
```
