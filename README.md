# Heart Disease Classification via Discriminant Analysis & Logistic Regression in R

**Author:** Tajwar Fahmid  
**Course:** ASDS 5303 – Statistical and Scientific Computing I (University of Texas at Arlington)  

## Project Overview
This project evaluates parametric and probabilistic classification methods—**Linear Discriminant Analysis (LDA)**, **Quadratic Discriminant Analysis (QDA)**, and **Logistic Regression**—to predict heart disease status (`Yes` vs. `No`) using clinical biomarkers (`Cholesterol` and `Resting_BP`) across a balanced cohort of $N = 200$ patients (`Patient_Data`).

The workflow covers stratified train/test partitioning, formal statistical assumption diagnostics (normality, homoscedasticity, multicollinearity, and Mahalanobis outlier detection), decision boundary visualization, clinical risk interpolation, probability cutoff optimization, and 5-fold cross-validation.

---

## Dataset & Experimental Setup
* **Dataset:** `Patient_Data` ($N = 200$ observations; $100$ `No`, $100$ `Yes`)
* **Predictors:**
  * `Cholesterol`: Serum cholesterol ($\text{mg/dL}$)
  * `Resting_BP`: Resting blood pressure ($\text{mm Hg}$)
* **Target:** `Disease_Status` (`No` = baseline, `Yes` = positive class)
* **Partitioning:** Stratified 80/20 train/test split (`set.seed(7435)`), yielding $160$ training patients ($80$ `No`, $80$ `Yes`) and $40$ unseen testing patients ($20$ `No`, $20$ `Yes`).

---

## Key Statistical Findings

### 1. Exploratory Data Analysis & LDA Diagnostics (Training Set, $n = 160$)
| Group | $n$ | Cholesterol Mean ± SD | Cholesterol Median (IQR) | Resting BP Mean ± SD | Resting BP Median (IQR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **No** | 80 | $191.63 \pm 19.06\text{ mg/dL}$ | $191.05\text{ (26.27)}$ | $126.28 \pm 10.79\text{ mm Hg}$ | $126.20\text{ (15.45)}$ |
| **Yes** | 80 | $244.15 \pm 26.40\text{ mg/dL}$ | $247.65\text{ (28.62)}$ | $149.74 \pm 15.71\text{ mm Hg}$ | $148.15\text{ (22.05)}$ |

* **Within-Group Normality (Met):** Shapiro-Wilk tests confirmed univariate normality across all class-predictor pairs (`No` Cholesterol $p = 0.9339$, `No` Resting BP $p = 0.4317$, `Yes` Cholesterol $p = 0.3161$, `Yes` Resting BP $p = 0.7146$), supported by Q-Q plots.
* **Multicollinearity & Outliers (Met):** Within-group correlations were weak ($r_{\text{No}} = 0.2606$, $r_{\text{Yes}} = -0.1832$), and $0$ multivariate outliers exceeded the $\chi^2_{2, 0.999} = 13.82$ Mahalanobis distance threshold.
* **Equal Covariance Matrices (Violated):** The `Yes` group displayed roughly double the variance in both predictors (`Cholesterol` $696.74$ vs. $363.46$; `Resting_BP` $246.85$ vs. $116.41$) with an inverted covariance direction ($-75.98$ vs. $+53.61$). **Box's M test** formally rejected homoscedasticity ($\text{Box's M} = 27.17$, $\chi^2 = 26.80$, $df = 3$, $p < 0.0001$), theoretically favoring QDA over LDA.

---

### 2. Model Comparison on Held-Out Test Set ($n = 40$)
Standardized LDA discriminant coefficients showed that **Cholesterol** ($\text{LD1} = 1.2194$) contributes more strongly to group separation than **Resting BP** ($\text{LD1} = 0.8280$). All three classifiers substantially outperformed the $50.0\%$ majority-class baseline and achieved identical classification metrics at the $0.50$ probability threshold:

| Metric | Majority Baseline | LDA | QDA | Logistic Regression ($p \ge 0.50$) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 50.00% | **92.50%** | **92.50%** | **92.50%** |
| **Sensitivity (Recall)** | — | **95.00%** | **95.00%** | **95.00%** |
| **Specificity** | — | **90.00%** | **90.00%** | **90.00%** |
| **Precision (PPV)** | — | **90.48%** | **90.48%** | **90.48%** |
| **Class `'No'` Accuracy** | — | 90.00% | 90.00% | 90.00% |
| **Class `'Yes'` Accuracy** | — | 95.00% | 95.00% | 95.00% |

* **Why LDA, QDA, and Logistic Regression Agree:** While QDA curves tightly around the tighter `No` cluster in the outer tails, all three decision boundaries pass through the exact same central transition zone ($\text{Cholesterol } 200\text{--}230\text{ mg/dL}$, $\text{Resting BP } 125\text{--}148\text{ mm Hg}$) where the test observations sit, misclassifying the same 3 borderline patients ($1$ false negative, $2$ false positives).

---

### 3. Clinical Interpolation Case Study: New Patient `(220 mg/dL, 140 mm Hg)`
* **Interpolation Check:** Both measurements fall well inside observed training bounds (`Cholesterol`: $148.4\text{--}313.2\text{ mg/dL}$; `Resting_BP`: $105.8\text{--}186.8\text{ mm Hg}$).
* **Model Predictions:**
  * **LDA:** Predicts **`Yes`** ($P(\text{Yes}) = 0.6188$, $P(\text{No}) = 0.3812$)
  * **QDA:** Predicts **`Yes`** ($P(\text{Yes}) = 0.5168$, $P(\text{No}) = 0.4832$)
* **Interpretation:** QDA's posterior probability lies much closer to the $50\%$ boundary because its quadratic curve wraps around the `No` cluster, flagging `(220, 140)` as a borderline risk profile.

---

### 4. Logistic Regression Odds Ratios, Cutoff Analysis & 5-Fold CV
* **Odds Ratios:** Each $1\text{ mg/dL}$ increase in Cholesterol increases heart disease odds by **$8.92\%$** ($\text{OR} = 1.0892$, $p < 0.0001$), and each $1\text{ mm Hg}$ increase in Resting BP increases odds by **$13.00\%$** ($\text{OR} = 1.1300$, $p < 0.0001$).
* **Threshold Trade-Off:** Adjusting the probability cutoff across `0.30`, `0.50`, and `0.70` shifts specificity from $80.0\% \to 90.0\% \to 95.0\%$ while holding sensitivity at $95.0\%$ on the test split. Clinically, a lower threshold (`0.30`) is preferred to minimize life-threatening false negatives.
* **5-Fold Cross-Validation ($N = 200$):** Out-of-fold CV accuracy is **$91.50\%$** (Sensitivity = $93.00\%$, Specificity = $90.00\%$, Mean ROC AUC = $0.9775$), providing a lower-variance generalization estimate than the single 40-patient test split ($92.50\%$).

---

## Requirements & How to Run

### R Packages Required
```r
install.packages(c("readxl", "dplyr", "ggplot2", "caret", "MASS", "heplots"))
