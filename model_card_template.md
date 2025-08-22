# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
**Model Type:** RandomForestClassifier
**Training Data:** Census data including features such as **age**, **workclass**, **education**, **martial status**, **occupation**, **relationship**, **race**, **sex**, **hours-per-week**, and **native-country**.
**Version:** 1.0

## Intended Use
Classify whether an individual's salary is above or below $50K based on demographic and employment features.

## Training Data
Data from the US Census inclusing categorical and continuous features as listed above. The dataset was spit into **80% training** and **20% testing**.

## Evaluation Data
Evaluation was performed on the **test split** of the Census dataset. The model performance was evaluated on **slices of categorical features** to detect potential disparities.

## Metrics
**Overall metrics:**
-**Percision:** 0.7419
=**Recall:** 0.6384
-**F1 Score:** 0.6863

> Note: Full slice metrics can be found in `slice_output.txt`.

## Ethical Considerations
The model may reflect historical biases in the census data (like race, sex, or workclass) since it was from 1994.

## Caveats and Recommendations
Do not rely on this model for high-stakes decisions. Use it mainly for **analysis, research or learning**. Remember to check **slice metrics** before trusting predicions and always treat predictions as **informative, not absolute**.