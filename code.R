
######## Mini Project 5: Discriminant Analysis
######## By Tajwar Fahmid

## Part 2: Data Exploration
# Load required packages
library(readxl)
library(dplyr)
library(ggplot2)
library(caret)

# Dataset Loaded as Patient_Data
Patient_Data

# 2.1. Splitting data:
set.seed(7435)

# 80/20 split
train_idx <- createDataPartition(Patient_Data$Disease_Status, p = 0.80, list = FALSE)
train_set <- Patient_Data[train_idx, ]
test_set  <- Patient_Data[-train_idx, ]

cat("=== Training Set (80%) ===\n")
cat("Total Patients:", nrow(train_set), "\n")
print(table(train_set$Disease_Status))

cat("\n=== Testing Set (20%) ===\n")
cat("Total Patients:", nrow(test_set), "\n")
print(table(test_set$Disease_Status))

# Splitting the data into two sets is important as the train_set is used to train the model. 
# And the test_set is used to check if the model is performing upto performance or not.

# 2.2. Training Set Summary Statistics:
train_summary <- train_set %>%
  group_by(Disease_Status) %>%
  summarise(
    Count        = n(),
    Chol_Mean    = round(mean(Cholesterol), 2),
    Chol_SD      = round(sd(Cholesterol), 2),
    Chol_Median  = round(median(Cholesterol), 2),
    Chol_IQR     = round(IQR(Cholesterol), 2),
    Chol_Min     = min(Cholesterol),
    Chol_Max     = max(Cholesterol),
    BP_Mean      = round(mean(Resting_BP), 2),
    BP_SD        = round(sd(Resting_BP), 2),
    BP_Median    = round(median(Resting_BP), 2),
    BP_IQR       = round(IQR(Resting_BP), 2),
    BP_Min       = min(Resting_BP),
    BP_Max       = max(Resting_BP)
  )

print(as.data.frame(train_summary))

# 2.3. Training Set Scatterplot:
p_scatter <- ggplot(train_set, aes(x = Cholesterol, y = Resting_BP, color = Disease_Status)) +
  geom_point(size = 3, alpha = 0.85) +
  scale_color_manual(values = c("No" = "#1f77b4", "Yes" = "#d62728")) +
  labs(
    title = "Cholesterol vs. Resting Blood Pressure (Training Set)",
    x = "Serum Cholesterol (mg/dL)",
    y = "Resting Blood Pressure (mm Hg)",
    color = "Heart Disease"
  ) +
  theme_bw(base_size = 12)

print(p_scatter)

# 2.4. The groups are reasonably separable as patients with no heart disease can be seen to have significantly lower levels of cholesterol and resting blood pressure.

# 2.5. LDA Assumptions:
# Ensure Disease_Status is a factor and convert train_set to a base data.frame
train_df <- as.data.frame(train_set)
train_df$Disease_Status <- factor(train_df$Disease_Status, levels = c("No", "Yes"))

# 5a. Normality of each predictor within each group (Shapiro-Wilk tests)
no_train  <- subset(train_df, Disease_Status == "No")
yes_train <- subset(train_df, Disease_Status == "Yes")

cat("=== Shapiro-Wilk Normality Tests ===\n")
cat("No  - Cholesterol p-value:", round(shapiro.test(no_train$Cholesterol)$p.value, 4), "\n")
cat("No  - Resting_BP p-value: ", round(shapiro.test(no_train$Resting_BP)$p.value, 4), "\n")
cat("Yes - Cholesterol p-value:", round(shapiro.test(yes_train$Cholesterol)$p.value, 4), "\n")
cat("Yes - Resting_BP p-value: ", round(shapiro.test(yes_train$Resting_BP)$p.value, 4), "\n")

# Visual check with Q-Q plots
par(mfrow = c(2, 2))
qqnorm(no_train$Cholesterol,  main = "Q-Q: Cholesterol (No)");  qqline(no_train$Cholesterol,  col = "blue")
qqnorm(no_train$Resting_BP,   main = "Q-Q: Resting BP (No)");   qqline(no_train$Resting_BP,   col = "blue")
qqnorm(yes_train$Cholesterol, main = "Q-Q: Cholesterol (Yes)"); qqline(yes_train$Cholesterol, col = "red")
qqnorm(yes_train$Resting_BP,  main = "Q-Q: Resting BP (Yes)");  qqline(yes_train$Resting_BP,  col = "red")
par(mfrow = c(1, 1))

# 5b. Equality of group covariance matrices & Box's M test
cat("\n=== Group Covariance Matrices ===\n")
S_no  <- cov(no_train[, c("Cholesterol", "Resting_BP")])
S_yes <- cov(yes_train[, c("Cholesterol", "Resting_BP")])
cat("Covariance Matrix ('No'):\n");  print(S_no)
cat("\nCovariance Matrix ('Yes'):\n"); print(S_yes)

cat("\n=== Box's M Test for Equal Covariances ===\n")
if (requireNamespace("heplots", quietly = TRUE)) {
  print(heplots::boxM(train_df[, c("Cholesterol", "Resting_BP")], train_df$Disease_Status))
} else {
  # Base R Box's M test (no external packages needed)
  n1 <- nrow(no_train); n2 <- nrow(yes_train); p <- 2; k <- 2
  S_pool <- ((n1 - 1) * S_no + (n2 - 1) * S_yes) / (n1 + n2 - k)
  M <- (n1 + n2 - k) * log(det(S_pool)) - ((n1 - 1) * log(det(S_no)) + (n2 - 1) * log(det(S_yes)))
  c_factor <- (2 * p^2 + 3 * p - 1) / (6 * (p + 1) * (k - 1)) * (1/(n1 - 1) + 1/(n2 - 1) - 1/(n1 + n2 - k))
  chi_sq <- M * (1 - c_factor)
  df_box <- p * (p + 1) * (k - 1) / 2
  p_val <- pchisq(chi_sq, df = df_box, lower.tail = FALSE)
  cat(sprintf("Box's M = %.4f, Chi-Sq = %.4f, df = %d, p-value = %.4f\n", M, chi_sq, df_box, p_val))
}

# 5c. Correlation between Cholesterol and Resting_BP (Overall and Within-Group)
cat("\n=== Predictor Correlations ===\n")
cat("Overall Correlation:", round(cor(train_df$Cholesterol, train_df$Resting_BP), 4), "\n")
cat("Within 'No' Corr:   ", round(cor(no_train$Cholesterol, no_train$Resting_BP), 4), "\n")
cat("Within 'Yes' Corr:  ", round(cor(yes_train$Cholesterol, yes_train$Resting_BP), 4), "\n")

# 5d. Outlier Check via Within-Group Mahalanobis Distance (chi-square df=2, p < 0.001 cutoff)
cutoff_mahal <- qchisq(0.999, df = 2)

mahal_no  <- mahalanobis(no_train[, c("Cholesterol", "Resting_BP")],
                         colMeans(no_train[, c("Cholesterol", "Resting_BP")]), S_no)

mahal_yes <- mahalanobis(yes_train[, c("Cholesterol", "Resting_BP")],
                         colMeans(yes_train[, c("Cholesterol", "Resting_BP")]), S_yes)

cat("\n=== Multivariate Outlier Check (Mahalanobis Distance > 13.82) ===\n")
cat("Outliers in 'No' group: ", sum(mahal_no > cutoff_mahal), "\n")
cat("Outliers in 'Yes' group:", sum(mahal_yes > cutoff_mahal), "\n")

## Part 3: Linear Discriminant Analysis

library(MASS)
library(ggplot2)
library(caret)

# Ensure Disease_Status is a factor in both train_set and test_set
train_set$Disease_Status <- factor(train_set$Disease_Status, levels = c("No", "Yes"))
test_set$Disease_Status  <- factor(test_set$Disease_Status,  levels = c("No", "Yes"))

# --- 3.1 Fit LDA Model & Standardized Coefficients ---
lda_fit <- lda(Disease_Status ~ Cholesterol + Resting_BP, data = train_set)
cat("=== Raw LDA Model Output ===\n")
print(lda_fit)

# Standardize predictors on training set so coefficients are on the same scale (mean = 0, SD = 1)
train_std <- train_set
train_std$Cholesterol <- scale(train_set$Cholesterol)
train_std$Resting_BP  <- scale(train_set$Resting_BP)

lda_std <- lda(Disease_Status ~ Cholesterol + Resting_BP, data = train_std)
cat("\n=== Standardized LD1 Coefficients (Comparable Scale) ===\n")
print(lda_std$scaling)


# --- 3.2 Prior Probabilities & Majority-Class Baseline (Test Set) ---
cat("\n=== Group Prior Probabilities ===\n")
print(lda_fit$prior)

test_counts <- table(test_set$Disease_Status)
cat("\nTest Set Class Counts:\n")
print(test_counts)

majority_baseline <- max(test_counts) / sum(test_counts)
cat("Majority-Class (Do-Nothing) Baseline Accuracy on Test Set:", round(majority_baseline, 4), "\n")


# --- 3.3 Plot LDA Decision Boundary on the Training Scatterplot ---
# Create a fine grid across the predictor space
grid_chol <- seq(min(train_set$Cholesterol) - 10, max(train_set$Cholesterol) + 10, length.out = 300)
grid_bp   <- seq(min(train_set$Resting_BP) - 10,  max(train_set$Resting_BP) + 10,  length.out = 300)
grid_df   <- expand.grid(Cholesterol = grid_chol, Resting_BP = grid_bp)

# Predict posterior probability of "Yes" across the grid
grid_df$prob_lda <- predict(lda_fit, newdata = grid_df)$posterior[, "Yes"]

p_lda_boundary <- ggplot() +
  geom_point(data = train_set, aes(x = Cholesterol, y = Resting_BP, color = Disease_Status),
             size = 3, alpha = 0.85) +
  geom_contour(data = grid_df, aes(x = Cholesterol, y = Resting_BP, z = prob_lda),
               breaks = 0.5, color = "black", linewidth = 1.2) +
  scale_color_manual(values = c("No" = "#1f77b4", "Yes" = "#d62728")) +
  labs(
    title = "LDA Decision Boundary on Training Set (P(Yes) = 0.5)",
    x = "Serum Cholesterol (mg/dL)",
    y = "Resting Blood Pressure (mm Hg)",
    color = "Heart Disease"
  ) +
  theme_bw(base_size = 12)

print(p_lda_boundary)


# --- 3.4 Test Set Confusion Matrix & Evaluation Metrics ---
lda_test_pred <- predict(lda_fit, newdata = test_set)

# Generate confusion matrix with "Yes" as the positive class
lda_cm <- confusionMatrix(lda_test_pred$class, test_set$Disease_Status, positive = "Yes")

cat("\n=== LDA Test Set Confusion Matrix ===\n")
print(lda_cm$table)

cat("\n=== Overall & Positive-Class ('Yes') Metrics ===\n")
cat("Accuracy:   ", round(lda_cm$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(lda_cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(lda_cm$byClass["Specificity"], 4), "\n")
cat("Precision:  ", round(lda_cm$byClass["Pos Pred Value"], 4), "\n")

# Per-class accuracies
acc_no  <- lda_cm$table["No", "No"]   / sum(lda_cm$table[, "No"])
acc_yes <- lda_cm$table["Yes", "Yes"] / sum(lda_cm$table[, "Yes"])

cat("\n=== Per-Class Accuracy ===\n")
cat("Class 'No' Accuracy: ", round(acc_no, 4), "\n")
cat("Class 'Yes' Accuracy:", round(acc_yes, 4), "\n")

## Part 4: Quadratic Discriminant Analysis
# ==============================================================================
library(MASS)
library(ggplot2)
library(caret)

# --- 4.1 Fit QDA Model on Training Set ---
qda_fit <- qda(Disease_Status ~ Cholesterol + Resting_BP, data = train_set)
cat("=== QDA Model Output ===\n")
print(qda_fit)


# --- 4.2 Confusion Matrix & Evaluation Metrics on Test Set Only ---
qda_test_pred <- predict(qda_fit, newdata = test_set)

# Generate confusion matrix with "Yes" as the positive class
qda_cm <- confusionMatrix(qda_test_pred$class, test_set$Disease_Status, positive = "Yes")

cat("\n=== QDA Test Set Confusion Matrix ===\n")
print(qda_cm$table)

cat("\n=== Overall & Positive-Class ('Yes') Metrics (QDA) ===\n")
cat("Accuracy:   ", round(qda_cm$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(qda_cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(qda_cm$byClass["Specificity"], 4), "\n")
cat("Precision:  ", round(qda_cm$byClass["Pos Pred Value"], 4), "\n")

# Per-class accuracies
acc_no_qda  <- qda_cm$table["No", "No"]   / sum(qda_cm$table[, "No"])
acc_yes_qda <- qda_cm$table["Yes", "Yes"] / sum(qda_cm$table[, "Yes"])

cat("\n=== Per-Class Accuracy (QDA) ===\n")
cat("Class 'No' Accuracy: ", round(acc_no_qda, 4), "\n")
cat("Class 'Yes' Accuracy:", round(acc_yes_qda, 4), "\n")


# --- 4.3 Side-by-Side Comparison Table (LDA vs. QDA on Test Set) ---
comparison_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity (Recall)", "Specificity", "Precision", "Class 'No' Accuracy", "Class 'Yes' Accuracy"),
  LDA = round(c(
    lda_cm$overall["Accuracy"],
    lda_cm$byClass["Sensitivity"],
    lda_cm$byClass["Specificity"],
    lda_cm$byClass["Pos Pred Value"],
    lda_cm$table["No", "No"] / sum(lda_cm$table[, "No"]),
    lda_cm$table["Yes", "Yes"] / sum(lda_cm$table[, "Yes"])
  ), 4),
  QDA = round(c(
    qda_cm$overall["Accuracy"],
    qda_cm$byClass["Sensitivity"],
    qda_cm$byClass["Specificity"],
    qda_cm$byClass["Pos Pred Value"],
    acc_no_qda,
    acc_yes_qda
  ), 4)
)
cat("\n=== LDA vs. QDA Test Set Performance Comparison ===\n")
print(comparison_df, row.names = FALSE)


# --- 4.4 Overlay QDA & LDA Decision Boundaries on Scatterplot ---
# Re-verify grid exists from Part 3 and compute QDA posterior probabilities
grid_chol <- seq(min(train_set$Cholesterol) - 10, max(train_set$Cholesterol) + 10, length.out = 300)
grid_bp   <- seq(min(train_set$Resting_BP) - 10,  max(train_set$Resting_BP) + 10,  length.out = 300)
grid_df   <- expand.grid(Cholesterol = grid_chol, Resting_BP = grid_bp)

grid_df$prob_lda <- predict(lda_fit, newdata = grid_df)$posterior[, "Yes"]
grid_df$prob_qda <- predict(qda_fit, newdata = grid_df)$posterior[, "Yes"]

p_overlay <- ggplot() +
  geom_point(data = train_set, aes(x = Cholesterol, y = Resting_BP, color = Disease_Status),
             size = 3, alpha = 0.8) +
  geom_contour(data = grid_df, aes(x = Cholesterol, y = Resting_BP, z = prob_lda, linetype = "LDA (Linear)"),
               breaks = 0.5, color = "black", linewidth = 1.2) +
  geom_contour(data = grid_df, aes(x = Cholesterol, y = Resting_BP, z = prob_qda, linetype = "QDA (Quadratic)"),
               breaks = 0.5, color = "darkviolet", linewidth = 1.2) +
  scale_color_manual(values = c("No" = "#1f77b4", "Yes" = "#d62728")) +
  scale_linetype_manual(values = c("LDA (Linear)" = "solid", "QDA (Quadratic)" = "dashed")) +
  labs(
    title = "Decision Boundaries: LDA vs. QDA (Training Set)",
    subtitle = "Boundary plotted at P(Disease_Status = Yes) = 0.5",
    x = "Serum Cholesterol (mg/dL)",
    y = "Resting Blood Pressure (mm Hg)",
    color = "Heart Disease",
    linetype = "Classifier"
  ) +
  theme_bw(base_size = 12)

print(p_overlay)

## Part 5: LDA in practice
# Create a data frame for the new patient (Cholesterol = 220, Resting_BP = 140)
new_patient <- data.frame(Cholesterol = 220, Resting_BP = 140)

# --- 5.1 Classify New Patient Using Fitted LDA Model ---
lda_new_pred <- predict(lda_fit, newdata = new_patient)

cat("=== 5.1 LDA Prediction for New Patient (Cholesterol = 220, Resting_BP = 140) ===\n")
cat("Predicted Class:       ", as.character(lda_new_pred$class), "\n")
cat("Posterior Prob ('No'): ", round(lda_new_pred$posterior[1, "No"], 4), "\n")
cat("Posterior Prob ('Yes'):", round(lda_new_pred$posterior[1, "Yes"], 4), "\n")


# --- 5.2 Classify Same Patient Using Fitted QDA Model & Check Observed Range ---
qda_new_pred <- predict(qda_fit, newdata = new_patient)

cat("\n=== 5.2 QDA Prediction for New Patient (Cholesterol = 220, Resting_BP = 140) ===\n")
cat("Predicted Class:       ", as.character(qda_new_pred$class), "\n")
cat("Posterior Prob ('No'): ", round(qda_new_pred$posterior[1, "No"], 4), "\n")
cat("Posterior Prob ('Yes'):", round(qda_new_pred$posterior[1, "Yes"], 4), "\n")

# Side-by-side comparison table
new_patient_comp <- data.frame(
  Model           = c("LDA", "QDA"),
  Predicted_Class = c(as.character(lda_new_pred$class), as.character(qda_new_pred$class)),
  Prob_No         = round(c(lda_new_pred$posterior[1, "No"],  qda_new_pred$posterior[1, "No"]), 4),
  Prob_Yes        = round(c(lda_new_pred$posterior[1, "Yes"], qda_new_pred$posterior[1, "Yes"]), 4)
)
cat("\n=== LDA vs. QDA Comparison for (220, 140) ===\n")
print(new_patient_comp, row.names = FALSE)

# Verify whether (220, 140) lies inside the observed range of the training/full data
cat("\n=== Observed Data Bounds ===\n")
cat("Training Cholesterol Range:", round(min(train_set$Cholesterol), 2), "to", round(max(train_set$Cholesterol), 2), "\n")
cat("Training Resting_BP Range: ", round(min(train_set$Resting_BP), 2),  "to", round(max(train_set$Resting_BP), 2), "\n")

inside_range <- (220 >= min(train_set$Cholesterol) & 220 <= max(train_set$Cholesterol)) &
  (140 >= min(train_set$Resting_BP)  & 140 <= max(train_set$Resting_BP))
cat("Is (220, 140) inside the observed predictor range?:", inside_range, "\n")

# ==============================================================================
# Bonus: Logistic Regression
# ==============================================================================
library(caret)

# Ensure Disease_Status is a factor with "No" as reference and "Yes" as positive
Patient_Data$Disease_Status <- factor(Patient_Data$Disease_Status, levels = c("No", "Yes"))
train_set$Disease_Status    <- factor(train_set$Disease_Status,    levels = c("No", "Yes"))
test_set$Disease_Status     <- factor(test_set$Disease_Status,     levels = c("No", "Yes"))

# --- Bonus 1: Fit Logistic Regression Model & Interpret Coefficients ---
logit_fit <- glm(Disease_Status ~ Cholesterol + Resting_BP, data = train_set, family = binomial)
cat("=== Bonus 1: Logistic Regression Model Summary ===\n")
print(summary(logit_fit))

cat("\n=== Odds Ratios (exp(Coefficients)) ===\n")
print(round(exp(coef(logit_fit)), 4))


# --- Bonus 2: Test Set Confusion Matrix & Evaluation (Cutoff = 0.5) ---
test_prob_logit <- predict(logit_fit, newdata = test_set, type = "response")
test_pred_50    <- factor(ifelse(test_prob_logit >= 0.5, "Yes", "No"), levels = c("No", "Yes"))

logit_cm_50 <- confusionMatrix(test_pred_50, test_set$Disease_Status, positive = "Yes")

cat("\n=== Bonus 2: Logistic Regression Test Confusion Matrix (Cutoff = 0.5) ===\n")
print(logit_cm_50$table)

cat("\n=== Overall & Positive-Class ('Yes') Metrics (Cutoff = 0.5) ===\n")
cat("Accuracy:   ", round(logit_cm_50$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(logit_cm_50$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(logit_cm_50$byClass["Specificity"], 4), "\n")
cat("Precision:  ", round(logit_cm_50$byClass["Pos Pred Value"], 4), "\n")

# Per-class accuracies
acc_no_logit  <- logit_cm_50$table["No", "No"]   / sum(logit_cm_50$table[, "No"])
acc_yes_logit <- logit_cm_50$table["Yes", "Yes"] / sum(logit_cm_50$table[, "Yes"])

cat("\n=== Per-Class Accuracy (Logistic Regression) ===\n")
cat("Class 'No' Accuracy: ", round(acc_no_logit, 4), "\n")
cat("Class 'Yes' Accuracy:", round(acc_yes_logit, 4), "\n")


# --- Bonus 3: Compare Logistic Regression vs. LDA on Test Set ---
logit_vs_lda <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision", "Class 'No' Accuracy", "Class 'Yes' Accuracy"),
  LDA = round(c(
    lda_cm$overall["Accuracy"],
    lda_cm$byClass["Sensitivity"],
    lda_cm$byClass["Specificity"],
    lda_cm$byClass["Pos Pred Value"],
    lda_cm$table["No", "No"] / sum(lda_cm$table[, "No"]),
    lda_cm$table["Yes", "Yes"] / sum(lda_cm$table[, "Yes"])
  ), 4),
  Logistic_Reg = round(c(
    logit_cm_50$overall["Accuracy"],
    logit_cm_50$byClass["Sensitivity"],
    logit_cm_50$byClass["Specificity"],
    logit_cm_50$byClass["Pos Pred Value"],
    acc_no_logit,
    acc_yes_logit
  ), 4)
)
cat("\n=== Bonus 3: LDA vs. Logistic Regression Comparison ===\n")
print(logit_vs_lda, row.names = FALSE)


# --- Bonus 4: Cutoff Analysis (0.3, 0.5, 0.7) ---
cutoffs <- c(0.3, 0.5, 0.7)
cutoff_table <- data.frame(
  Cutoff      = cutoffs,
  Sensitivity = numeric(length(cutoffs)),
  Specificity = numeric(length(cutoffs)),
  Precision   = numeric(length(cutoffs)),
  Accuracy    = numeric(length(cutoffs))
)

for (i in seq_along(cutoffs)) {
  c_val  <- cutoffs[i]
  c_pred <- factor(ifelse(test_prob_logit >= c_val, "Yes", "No"), levels = c("No", "Yes"))
  cm_c   <- confusionMatrix(c_pred, test_set$Disease_Status, positive = "Yes")
  
  if (c_val %in% c(0.3, 0.7)) {
    cat(sprintf("\n--- Confusion Matrix at Cutoff = %.1f ---\n", c_val))
    print(cm_c$table)
  }
  
  cutoff_table$Sensitivity[i] <- round(cm_c$byClass["Sensitivity"], 4)
  cutoff_table$Specificity[i] <- round(cm_c$byClass["Specificity"], 4)
  cutoff_table$Precision[i]   <- round(cm_c$byClass["Pos Pred Value"], 4)
  cutoff_table$Accuracy[i]    <- round(cm_c$overall["Accuracy"], 4)
}

cat("\n=== Bonus 4: Sensitivity vs. Specificity Across Cutoffs ===\n")
print(cutoff_table, row.names = FALSE)


# --- Bonus 5: 5-Fold Cross-Validation (Using All 200 Patients in Patient_Data) ---
set.seed(7435)
cv_ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

cv_logit <- train(
  Disease_Status ~ Cholesterol + Resting_BP,
  data      = Patient_Data,
  method    = "glm",
  family    = binomial,
  metric    = "ROC",
  trControl = cv_ctrl
)

# Compute overall 5-fold out-of-fold accuracy and confusion matrix
cv_cm <- confusionMatrix(cv_logit$pred$pred, cv_logit$pred$obs, positive = "Yes")

cat("\n=== Bonus 5: 5-Fold Cross-Validation Results (n = 200) ===\n")
print(cv_logit$results)
cat("5-Fold CV Out-of-Fold Accuracy:   ", round(cv_cm$overall["Accuracy"], 4), "\n")
cat("5-Fold CV Out-of-Fold Sensitivity:", round(cv_cm$byClass["Sensitivity"], 4), "\n")
cat("5-Fold CV Out-of-Fold Specificity:", round(cv_cm$byClass["Specificity"], 4), "\n")
cat("Single 80/20 Split Accuracy:      ", round(logit_cm_50$overall["Accuracy"], 4), "\n")
