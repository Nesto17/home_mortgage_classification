# Loading required libraries
library(tidyverse)
library(tidymodels)

library(rpart)
library(ranger)
library(baguette)
library(bonsai)
library(lightgbm)
library(xgboost)
library(keras)
library(tensorflow)
library(nnet)
library(reticulate)
library(mda)
library(discrim)
library(stacks)

# Reading the datasets
train <- read_csv("train2.csv")
test <- read_csv("test2.csv")

# Removing problematic features
train <- recipe(action_taken ~ ., data = train) %>%
  step_rm(id, age_of_co_applicant_or_co_borrower) %>% 
  step_filter_missing(all_predictors(), threshold = 0.1) %>% 
  step_nzv(all_predictors()) %>% 
  prep() %>% 
  juice()

# Removing numerical outliers
remove_outliers <- function(df) {
  filter <- rep(TRUE, nrow(df))
  
  for (column_name in names(df)) {
    if (is.numeric(df[[column_name]])) {
      P_1 <- quantile(df[[column_name]], 0.005, na.rm = TRUE)
      P_2 <- quantile(df[[column_name]], 0.995, na.rm = TRUE)
      
      filter <- filter & (df[[column_name]] >= P_1 & df[[column_name]] <= P_2 | is.na(df[[column_name]]))
    }
  }
  df <- df[filter, ]
  return(df)
}
train <- remove_outliers(train)

# Converting numerical to factors
factor_conversion <- function(feat) {
  if (n_distinct(feat) <= 20) {
    feat <- as.factor(feat)
  }
  return(feat)
}
train <- train %>%
  mutate(across(everything(), factor_conversion))
test <- test %>%
  mutate(across(everything(), factor_conversion))


# ------------------------------------------------------------


# Creating 10 cross-validation folds
set.seed(2023)
folds <- vfold_cv(data = train, v = 10, strata = action_taken)

# Preprocessing recipe
rec <- recipe(action_taken ~ ., data = train) %>% 
  step_mutate(state = factor(state)) %>% 
  step_mutate(state = if_else(state == "PR", NA, state)) %>% 
  step_mutate(age_of_applicant_or_borrower = if_else(
    age_of_applicant_or_borrower %in% c(8888, 9999), 
    NA, 
    age_of_applicant_or_borrower)) %>% 
  step_impute_mean(all_numeric_predictors()) %>% 
  step_impute_mode(all_nominal_predictors()) %>%
  step_cut(loan_term, breaks = c(150, 210, 270, 330, 390), include_outside_range = TRUE) %>% 
  step_YeoJohnson(loan_amount, income, property_value) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) 


# ------------------------------------------------------------


# Model specifications
# Model 1 - Decision Tree
tree_spec <- decision_tree(
  tree_depth = 13,
  cost_complexity = 7.058241e-05,
  min_n = 26
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# Model 2 - Bagged Trees
bagtree_spec <- bag_tree(
  tree_depth = 9,
  min_n = 4,
  cost_complexity = 6.711050e-09
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# Model 3 - Boosted Trees
boosttree_spec <- boost_tree(
  trees = 1257,
  tree_depth = 8, 
  min_n = 11,
  learn_rate = 6.525540e-02
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

# Model 4 - Logistic Regression with Elastic-Net Regularization
logistic_spec <- logistic_reg(
  penalty = 2.989194e-08 , 
  mixture = 0.2268432
) %>%
  set_engine("glmnet") %>% 
  set_mode("classification")

# Model 5 - MLP (Single-Layer FNN)
mlp_spec <- mlp(
    epochs = 45,
    hidden_units = 8,
    penalty = 0.0000000366,
    activation = "softmax"
  ) %>%
  set_engine("keras") %>%
  set_mode("classification")

# Model 6 - Bagged MLP (Single-Layer FNN)
bagmlp_spec <- bag_mlp(
    hidden_units = 7,
    penalty = 1.18e-10,
    epochs = 30
  ) %>%
  set_engine("nnet") %>%
  set_mode("classification")

# Model 7 - Linear Discriminant Analysis
lda_spec <- discrim_linear() %>%
  set_engine("mda") %>%
  set_mode("classification")


# ------------------------------------------------------------


# Creating workflow
wfl <- workflow_set(
  preproc = list(mod = rec),
  models = list(tree = tree_spec,
                bagtree = bagtree_spec,
                boosttree = boosttree_spec,
                logistic = logistic_spec,
                mlp = mlp_spec,
                bagmlp = bagmlp_spec,
                lda = lda_spec),
  cross = TRUE
) %>% option_add(control = control_grid(save_workflow = TRUE, 
                                        save_pred = TRUE, verbose = TRUE))

# Fitting workflow
set.seed(2023)
wfl_fitted <- wfl %>% workflow_map("fit_resamples", 
                                   resamples = folds, verbose = TRUE)

# Constructing model stack
set.seed(2023)
model_stack <- stacks() %>% 
  add_candidates(wfl_fitted) %>% 
  blend_predictions() %>% 
  fit_members()


# ------------------------------------------------------------


# Making predictions
pred <- model_stack %>% 
  predict(test)

# Creating prediction tibble and export
pred <- test %>% 
  select(id) %>% 
  bind_cols(pred)

write_csv(pred, "pred.csv")




