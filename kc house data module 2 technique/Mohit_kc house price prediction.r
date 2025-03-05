# --------------------------------
# Load Required Libraries
# --------------------------------
library(ggplot2)
library(dplyr)
library(corrplot)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)

# --------------------------------
# Load the Dataset & Preprocessing
# --------------------------------
kc_data <- read.csv("kc_house_data.csv", stringsAsFactors = FALSE)

# Remove Duplicate Entries
kc_data <- kc_data[!duplicated(kc_data$id), ]

# Convert Zipcode to a Factor
kc_data$zipcode <- as.factor(kc_data$zipcode)

# Log Transformation of Price (to reduce skewness)
kc_data$log_price <- log(kc_data$price)

# --------------------------------
# Encode Zipcode into Price-Based Clusters (Fix for RandomForest)
# --------------------------------
zipcode_median_price <- kc_data %>%
  group_by(zipcode) %>%
  summarize(median_price = median(price)) %>%
  arrange(median_price)

# Assign Cluster Labels (Rank-Based Encoding)
zipcode_median_price$zipcode_cluster <- as.numeric(factor(zipcode_median_price$median_price, 
                                                          levels = unique(zipcode_median_price$median_price)))

# Merge the Clustered Zipcode Data Back into the Main Dataset
kc_data <- merge(kc_data, zipcode_median_price, by = "zipcode")

# Remove Original Zipcode Column (Since It's Categorical)
kc_data$zipcode <- NULL

# --------------------------------
# Train-Test Split (With Fixed Zipcode Encoding)
# --------------------------------
set.seed(123)
trainIndex <- createDataPartition(kc_data$log_price, p = 0.8, list = FALSE)
train_data <- kc_data[trainIndex, ]
test_data <- kc_data[-trainIndex, ]

# --------------------------------
# Enhanced Exploratory Data Analysis (EDA)
# --------------------------------

# 1. Enhanced Histogram: Log Price Distribution
ggplot(kc_data, aes(x = log_price)) +
  geom_histogram(binwidth = 0.1, fill = "purple", color = "black", alpha = 0.7) +
  labs(title = "Log-Transformed Price Distribution", x = "Log Price", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(size=14, face="bold"))

# 2. Enhanced Scatter Plot: Sqft Living vs. Price
ggplot(kc_data, aes(x = sqft_living, y = price)) +
  geom_point(color = "darkblue", alpha = 0.5) +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Scatter Plot: Sqft Living vs. Price", x = "Living Area (sqft)", y = "Price (USD)") +
  theme_minimal() +
  theme(plot.title = element_text(size=14, face="bold"))



## -------------------------------------------------------------------------------------------------------------------------

# Load Required Libraries (Ensure lubridate is loaded)
library(lubridate)

# Convert Date Column to Proper Date Format
kc_data$date <- as.Date(kc_data$date, format="%Y%m%d")

# Extract Month and Year for Seasonality Analysis
kc_data$month <- month(kc_data$date, label = TRUE)  # Month as Factor (Jan, Feb, etc.)
kc_data$year <- year(kc_data$date)

ggplot(kc_data, aes(x = month, y = price)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Seasonality Effects: Monthly Price Distribution", x = "Month", y = "Price (USD)") +
  theme_minimal()

ggplot(kc_data, aes(x = factor(year), y = price)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Yearly House Price Trends", x = "Year", y = "Price (USD)") +
  theme_minimal()

## -------------------------------------------------

# 3. Enhanced Correlation Heatmap
cor_matrix <- cor(kc_data[, c("price", "sqft_living", "bedrooms", "bathrooms", "grade", "sqft_lot")])
corrplot(cor_matrix, method = "number", type = "upper",
         title = "Correlation Heatmap of Key Variables", mar=c(0,0,1,0))

# --------------------------------
# Multiple Linear Regression Model (With Zipcode Clusters)
# --------------------------------
lm_model_zip <- lm(log_price ~ sqft_living + bedrooms + bathrooms + grade + zipcode_cluster, data = train_data)
summary(lm_model_zip)

# Predictions on Test Set
predictions_log_lm_zip <- predict(lm_model_zip, newdata = test_data)

# Convert Log Predictions Back to Price Scale
predictions_lm_zip <- exp(predictions_log_lm_zip)

# Evaluation Metrics
rmse_log_lm_zip <- sqrt(mean((test_data$price - predictions_lm_zip)^2))
mae_log_lm_zip <- mean(abs(test_data$price - predictions_lm_zip))

cat("Log-Transformed Linear Regression Model (with Zipcode Clusters) Evaluation:\n")
cat("RMSE:", rmse_log_lm_zip, "\n")
cat("MAE:", mae_log_lm_zip, "\n")

# --------------------------------
# Decision Tree Model (With Zipcode Clusters)
# --------------------------------
tree_model_zip <- rpart(log_price ~ sqft_living + bedrooms + bathrooms + grade + zipcode_cluster, data = train_data, method = "anova")
rpart.plot(tree_model_zip, main = "Decision Tree for Predicting House Prices (With Zipcode Clusters)")

# Predictions and Evaluation
predictions_tree_zip <- predict(tree_model_zip, newdata = test_data)
predictions_tree_zip <- exp(predictions_tree_zip)  # Convert back from log scale

rmse_tree_zip <- sqrt(mean((test_data$price - predictions_tree_zip)^2))
mae_tree_zip <- mean(abs(test_data$price - predictions_tree_zip))

cat("\nDecision Tree Model (With Zipcode Clusters) Evaluation:\n")
cat("RMSE:", rmse_tree_zip, "\n")
cat("MAE:", mae_tree_zip, "\n")


# --------------------------------
# Random Forest Model (With Zipcode Clusters)
# --------------------------------
rf_model_zip <- randomForest(log_price ~ sqft_living + bedrooms + bathrooms + grade + zipcode_cluster, 
                             data = train_data, ntree = 500, importance = TRUE)

# Predictions
predictions_rf_zip <- predict(rf_model_zip, newdata = test_data)

# Convert Log Predictions Back to Price Scale
predictions_rf_zip <- exp(predictions_rf_zip)

# Evaluation Metrics
rmse_rf_zip <- sqrt(mean((test_data$price - predictions_rf_zip)^2))
mae_rf_zip <- mean(abs(test_data$price - predictions_rf_zip))

cat("\nRandom Forest Model (With Zipcode Clusters) Evaluation:\n")
cat("RMSE:", rmse_rf_zip, "\n")
cat("MAE:", mae_rf_zip, "\n")

# --------------------------------
# Feature Importance from Random Forest
# --------------------------------
importance(rf_model_zip)
varImpPlot(rf_model_zip, main="Feature Importance in Random Forest Model (With Zipcode Clusters)")

