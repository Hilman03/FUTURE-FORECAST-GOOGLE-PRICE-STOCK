# =============================================================================
# STA572/570 - TIME SERIES ANALYSIS AND FORECASTING
# Assessment 3 (Group Project) - Section 5(a)
# Univariate Modelling Techniques for Google (GOOGL) Stock Price
# Dataset: Google Stock Daily Data (2013 - 2024)
# Models: Single Exponential Smoothing (SES), Holt's Method, ARRES
# =============================================================================

library(fpp2)
library(forecast)

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================

data_path <- file.path("data", "GoogleStock_Dataset.csv")
google <- read.csv(data_path)
google$Date <- as.Date(google$Date, format = "%d/%m/%Y")

cat("Ticker            : GOOGL (Google / Alphabet Inc.)\n")
cat("Total observations:", nrow(google), "\n")
cat("Date range        :", as.character(min(google$Date)),
    "to", as.character(max(google$Date)), "\n")
cat("Price range       : $", round(min(google$Close), 2),
    "to $", round(max(google$Close), 2), "\n")

# Convert Close to time series (trading days ~252 per year)
googlets <- ts(google$Close, frequency = 252)


# =============================================================================
# 2. DATA DESCRIPTION - TIME SERIES PLOT
# =============================================================================

autoplot(googlets) +
  ggtitle("GOOGL (Google) - Daily Closing Price") +
  ylab("Price (USD)") +
  xlab("Year") +
  theme_minimal()


# =============================================================================
# 3. FITTED / HOLD-OUT SPLIT  (70% fitted, 30% hold-out)
# =============================================================================

n        <- length(googlets)
split_pt <- round(n * 0.70)
est_part <- head(googlets, split_pt)
eva_part <- tail(googlets, n - split_pt)

cat("\nFitted part:", length(est_part),
    "| Hold-out part:", length(eva_part), "\n")


# =============================================================================
# 4. ERROR MEASURE FUNCTIONS
# =============================================================================

calc_MSE <- function(actual, fitted) {
  mean((actual - fitted)^2, na.rm = TRUE)
}

calc_MAPE <- function(actual, fitted) {
  mean(abs((actual - fitted) / actual) * 100, na.rm = TRUE)
}


# =============================================================================
# MODEL 1 : SINGLE EXPONENTIAL SMOOTHING (SES)
# -----------------------------------------------------------------------------
# Purpose : Captures the LEVEL of the series only. Suitable for data with
#           no clear trend or seasonality. Serves as a baseline benchmark.
# Initial value: l0 set to the first observation.
# Finding the parameter: calling ses() WITHOUT specifying alpha makes R search
#           for the alpha that minimises the sum of squared errors.
# =============================================================================

cat("\n============================================================\n")
cat("MODEL 1 : Single Exponential Smoothing (SES)\n")
cat("============================================================\n")

SE_est    <- ses(est_part, h = length(eva_part))
summary(SE_est)

alpha_ses <- SE_est$model$par["alpha"]

cat("Optimal Alpha (alpha):", round(alpha_ses, 4), "\n")
cat("Initial Level (l0)   :", round(SE_est$model$initstate, 4), "\n\n")

ses_fitted <- fitted(SE_est)

# Hold-out evaluation: apply the best alpha to the hold-out part and take the
# ONE-STEP-AHEAD fitted values. Using SE_est$mean instead would give a flat
# 833-day-ahead projection, which is not comparable with the other models.
ses_forecast <- fitted(ses(eva_part, alpha = alpha_ses))

ses_train_MSE  <- calc_MSE(est_part,  ses_fitted)
ses_train_MAPE <- calc_MAPE(est_part, ses_fitted)
ses_test_MSE   <- calc_MSE(eva_part,  ses_forecast)
ses_test_MAPE  <- calc_MAPE(eva_part, ses_forecast)

cat("--- Fitted Part Errors ---\n")
cat("MSE :", round(ses_train_MSE,  4), "\n")
cat("MAPE:", round(ses_train_MAPE, 4), "%\n\n")

cat("--- Hold-out Part Errors ---\n")
cat("MSE :", round(ses_test_MSE,  4), "\n")
cat("MAPE:", round(ses_test_MAPE, 4), "%\n\n")

ses_onestep <- forecast(ses(googlets, alpha = alpha_ses), h = 1)
cat("One-step-ahead Forecast (SES):", round(ses_onestep$mean, 2), "USD\n")

autoplot(googlets) +
  autolayer(ses_fitted,   series = "SES Fitted (fitted part)") +
  autolayer(ses_forecast, series = "SES Forecast (hold-out)") +
  ggtitle("Model 1: Single Exponential Smoothing - GOOGL") +
  ylab("Price (USD)") + xlab("Year") +
  theme_minimal()


# =============================================================================
# MODEL 2 : HOLT'S METHOD (Double Exponential Smoothing)
# -----------------------------------------------------------------------------
# Purpose : Extends SES by adding a TREND component (beta). GOOGL has shown a
#           strong long-term upward trend from $18 to $143 over 11 years,
#           making Holt's method a natural candidate.
# Initial values: l0 and b0 estimated by minimising SSE via optimisation.
# Finding the parameters: calling holt() without alpha and beta makes R search
#           for the pair that minimises the sum of squared errors.
# =============================================================================

cat("\n============================================================\n")
cat("MODEL 2 : Holt's Method (Double Exponential Smoothing)\n")
cat("============================================================\n")

Holt_est   <- holt(est_part, h = length(eva_part))
summary(Holt_est)

alpha_holt <- Holt_est$model$par["alpha"]
beta_holt  <- Holt_est$model$par["beta"]

cat("Optimal Alpha (alpha):", round(alpha_holt, 4), "\n")
cat("Optimal Beta  (beta) :", round(beta_holt,  4), "\n")
cat("Initial Level (l0)   :", round(Holt_est$model$initstate[1], 4), "\n")
cat("Initial Trend (b0)   :", round(Holt_est$model$initstate[2], 4), "\n\n")

holt_fitted <- fitted(Holt_est)

# Hold-out evaluation: one-step-ahead fitted values on the hold-out part, using
# the best alpha and beta (see the note in Model 1).
holt_forecast <- fitted(holt(eva_part, alpha = alpha_holt, beta = beta_holt))

holt_train_MSE  <- calc_MSE(est_part,  holt_fitted)
holt_train_MAPE <- calc_MAPE(est_part, holt_fitted)
holt_test_MSE   <- calc_MSE(eva_part,  holt_forecast)
holt_test_MAPE  <- calc_MAPE(eva_part, holt_forecast)

cat("--- Fitted Part Errors ---\n")
cat("MSE :", round(holt_train_MSE,  4), "\n")
cat("MAPE:", round(holt_train_MAPE, 4), "%\n\n")

cat("--- Hold-out Part Errors ---\n")
cat("MSE :", round(holt_test_MSE,  4), "\n")
cat("MAPE:", round(holt_test_MAPE, 4), "%\n\n")

holt_onestep <- forecast(holt(googlets, alpha = alpha_holt, beta = beta_holt), h = 1)
cat("One-step-ahead Forecast (Holt's):", round(holt_onestep$mean, 2), "USD\n")

autoplot(googlets) +
  autolayer(holt_fitted,   series = "Holt Fitted (fitted part)") +
  autolayer(holt_forecast, series = "Holt Forecast (hold-out)") +
  ggtitle("Model 2: Holt's Method - GOOGL") +
  ylab("Price (USD)") + xlab("Year") +
  theme_minimal()


# =============================================================================
# MODEL 3 : ADAPTIVE RESPONSE RATE EXPONENTIAL SMOOTHING (ARRES)
# -----------------------------------------------------------------------------
# Purpose : Self-adjusting model where alpha changes at each time step based
#           on recent forecast errors. Handles sudden stock price movements
#           (earnings releases, market crashes) automatically.
# Note    : There is NO "best parameter estimate" for ARRES - alpha adapts
#           dynamically at every time period.
# There is no built-in R function for ARRES, so the recursion from Chapter 2
# is coded manually:
#   e(t)     = y(t) - F(t)
#   E(t)     = beta * e(t)   + (1 - beta) * E(t-1)
#   AE(t)    = beta * |e(t)| + (1 - beta) * AE(t-1)
#   alpha(t) = | E(t) / AE(t) |
#   F(t+1)   = alpha(t) * y(t) + (1 - alpha(t)) * F(t)
# Initial values: F(1) = y(1), alpha(1) = 0.6, beta = 0.2
# =============================================================================

cat("\n============================================================\n")
cat("MODEL 3 : ARRES (Adaptive Response Rate Exponential Smoothing)\n")
cat("============================================================\n")

arres <- function(y, beta = 0.2, alpha1 = 0.6) {
  n     <- length(y)
  F     <- numeric(n)
  e     <- numeric(n)
  E     <- numeric(n)
  AE    <- numeric(n)
  alpha <- numeric(n)

  F[1]     <- y[1]
  alpha[1] <- alpha1
  E[1]     <- 0
  AE[1]    <- 0

  for (t in 2:n) {
    F[t]     <- alpha[t - 1] * y[t - 1] + (1 - alpha[t - 1]) * F[t - 1]
    e[t]     <- y[t] - F[t]
    E[t]     <- beta * e[t]      + (1 - beta) * E[t - 1]
    AE[t]    <- beta * abs(e[t]) + (1 - beta) * AE[t - 1]
    alpha[t] <- if (AE[t] == 0) alpha[t - 1] else abs(E[t] / AE[t])
  }

  list(fitted = F, alpha = alpha)
}

arres_train  <- arres(as.numeric(est_part))
arres_fitted <- arres_train$fitted

arres_train_MSE  <- calc_MSE(as.numeric(est_part),  arres_fitted)
arres_train_MAPE <- calc_MAPE(as.numeric(est_part), arres_fitted)

cat("Smoothing constant beta:", 0.2, "| Seed alpha1:", 0.6, "\n")
cat("Final alpha (last observation):", round(tail(arres_train$alpha, 1), 4), "\n\n")

cat("--- Fitted Part Errors ---\n")
cat("MSE :", round(arres_train_MSE,  4), "\n")
cat("MAPE:", round(arres_train_MAPE, 4), "%\n\n")

arres_full        <- arres(as.numeric(googlets))
arres_test_fitted <- tail(arres_full$fitted, length(eva_part))

arres_test_MSE  <- calc_MSE(as.numeric(eva_part),  arres_test_fitted)
arres_test_MAPE <- calc_MAPE(as.numeric(eva_part), arres_test_fitted)

cat("--- Hold-out Part Errors ---\n")
cat("MSE :", round(arres_test_MSE,  4), "\n")
cat("MAPE:", round(arres_test_MAPE, 4), "%\n\n")

last_alpha    <- tail(arres_full$alpha,  1)
last_fitted   <- tail(arres_full$fitted, 1)
last_actual   <- tail(as.numeric(googlets), 1)
arres_onestep <- last_alpha * last_actual + (1 - last_alpha) * last_fitted

cat("One-step-ahead Forecast (ARRES):", round(arres_onestep, 2), "USD\n")

arres_ts <- ts(arres_full$fitted, frequency = 252)
autoplot(googlets, series = "Actual") +
  autolayer(arres_ts, series = "ARRES Fitted") +
  ggtitle("Model 3: ARRES - GOOGL") +
  ylab("Price (USD)") + xlab("Year") +
  theme_minimal()


# =============================================================================
# 5. MODEL COMPARISON SUMMARY
# =============================================================================

cat("\n============================================================\n")
cat("MODEL COMPARISON SUMMARY - GOOGL\n")
cat("============================================================\n")

comparison <- data.frame(
  Model        = c("SES", "Holt's Method", "ARRES"),
  Train_MSE    = round(c(ses_train_MSE,  holt_train_MSE,  arres_train_MSE),  4),
  Train_MAPE   = round(c(ses_train_MAPE, holt_train_MAPE, arres_train_MAPE), 4),
  Holdout_MSE  = round(c(ses_test_MSE,   holt_test_MSE,   arres_test_MSE),   4),
  Holdout_MAPE = round(c(ses_test_MAPE,  holt_test_MAPE,  arres_test_MAPE),  4)
)

print(comparison)

best_MSE  <- comparison$Model[which.min(comparison$Holdout_MSE)]
best_MAPE <- comparison$Model[which.min(comparison$Holdout_MAPE)]

cat("\nBest model based on Hold-out MSE :", best_MSE,  "\n")
cat("Best model based on Hold-out MAPE:", best_MAPE, "\n")

# Comparison plot: all three fitted series against the actual data
ses_full_fitted  <- fitted(ses(googlets,  alpha = alpha_ses))
holt_full_fitted <- fitted(holt(googlets, alpha = alpha_holt, beta = beta_holt))

autoplot(googlets, series = "Actual") +
  autolayer(ses_full_fitted,  series = "SES") +
  autolayer(holt_full_fitted, series = "Holt") +
  autolayer(arres_ts,         series = "ARRES") +
  ggtitle("Model Comparison: Actual vs Fitted - GOOGL") +
  ylab("Price (USD)") + xlab("Year") +
  theme_minimal()

# -----------------------------------------------------------------------------
# EFFECTIVE COMPARISON 1 : ONE PANEL PER MODEL
# -----------------------------------------------------------------------------
# Each model gets its own panel against the grey actual price. Giving each model
# its own panel (instead of overlaying all three) keeps the comparison readable.
#
# n_show controls how many of the most recent days are shown:
#   n_show <- n_all   -> the WHOLE series (all days)
#   n_show <- 120     -> zoom into the last 120 days, where the one-day lag and
#                        the turning-point behaviour of each model are visible.
# On the full series the fitted lines lie almost on top of the actual price, so
# a zoom is still the clearest way to SEE the differences between the models.

n_all  <- length(googlets)
n_show <- n_all                          # show all days (set to 120 to zoom)
idx    <- (n_all - n_show + 1):n_all
dates  <- google$Date

cmp_df <- rbind(
  data.frame(Date = dates[idx], Price = as.numeric(googlets)[idx],         Series = "Actual"),
  data.frame(Date = dates[idx], Price = as.numeric(ses_full_fitted)[idx],  Series = "SES"),
  data.frame(Date = dates[idx], Price = as.numeric(holt_full_fitted)[idx], Series = "Holt"),
  data.frame(Date = dates[idx], Price = as.numeric(arres_ts)[idx],         Series = "ARRES")
)
cmp_df$Series <- factor(cmp_df$Series, levels = c("Actual", "SES", "Holt", "ARRES"))

# put the actual price alongside each model so it can be drawn as a grey backdrop
actual_only <- subset(cmp_df, Series == "Actual")[, c("Date", "Price")]
names(actual_only)[2] <- "Actual"
models_only <- merge(subset(cmp_df, Series != "Actual"), actual_only, by = "Date")

ggplot(models_only, aes(x = Date)) +
  geom_line(aes(y = Actual), colour = "grey65", linewidth = 0.8) +
  geom_line(aes(y = Price, colour = Series), linewidth = 0.4) +
  facet_wrap(~ Series, ncol = 1) +
  scale_colour_manual(values = c(SES = "#D55E00", Holt = "#0072B2", ARRES = "#009E73"),
                      guide = "none") +
  ggtitle("Univariate Models: Actual vs Fitted (full series)") +
  ylab("Price (USD)") + xlab(NULL) +
  theme_minimal()

# -----------------------------------------------------------------------------
# EFFECTIVE COMPARISON 2 : HOLD-OUT ERROR BAR CHART
# -----------------------------------------------------------------------------
# The clearest single image of which model is best: hold-out MAPE as bars,
# sorted from best to worst.

ggplot(comparison, aes(x = reorder(Model, Holdout_MAPE), y = Holdout_MAPE, fill = Model)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = round(Holdout_MAPE, 3)), vjust = -0.4, size = 3.5) +
  scale_fill_manual(values = c("SES" = "#D55E00", "Holt's Method" = "#0072B2",
                               "ARRES" = "#009E73"), guide = "none") +
  ggtitle("Hold-out MAPE by Univariate Model") +
  ylab("MAPE (%)") + xlab(NULL) +
  theme_minimal()

# -----------------------------------------------------------------------------
# EFFECTIVE COMPARISON 3 : ALL THREE MODELS OVER THE HOLD-OUT PERIOD
# -----------------------------------------------------------------------------
# Compares the one-step-ahead forecasts of the three models against the actual
# price over the hold-out part only (the 30% used for evaluation). This is the
# period on which the models are scored, so it is the fairest visual comparison.
# ses_forecast, holt_forecast and arres_test_fitted are the one-step-ahead
# forecasts on the hold-out part, already computed above.

hold_dates <- google$Date[(split_pt + 1):n]

hold_df <- rbind(
  data.frame(Date = hold_dates, Price = as.numeric(eva_part),          Series = "Actual"),
  data.frame(Date = hold_dates, Price = as.numeric(ses_forecast),      Series = "SES"),
  data.frame(Date = hold_dates, Price = as.numeric(holt_forecast),     Series = "Holt"),
  data.frame(Date = hold_dates, Price = as.numeric(arres_test_fitted), Series = "ARRES")
)
hold_df$Series <- factor(hold_df$Series, levels = c("Actual", "SES", "Holt", "ARRES"))

# --- version A: one panel per model against the actual price ------------------
hold_actual <- subset(hold_df, Series == "Actual")[, c("Date", "Price")]
names(hold_actual)[2] <- "Actual"
hold_models <- merge(subset(hold_df, Series != "Actual"), hold_actual, by = "Date")

ggplot(hold_models, aes(x = Date)) +
  geom_line(aes(y = Actual), colour = "grey65", linewidth = 0.5) +
  geom_line(aes(y = Price, colour = Series), linewidth = 0.4) +
  facet_wrap(~ Series, ncol = 1) +
  scale_colour_manual(values = c(SES = "#D55E00", Holt = "#0072B2", ARRES = "#009E73"),
                      guide = "none") +
  ggtitle("Univariate Models over the Hold-out Period") +
  ylab("Price (USD)") + xlab(NULL) +
  theme_minimal()

# --- version B: all three overlaid on one chart (with the actual price) -------
ggplot(hold_df, aes(x = Date, y = Price, colour = Series)) +
  geom_line(linewidth = 0.4) +
  scale_colour_manual(values = c(Actual = "grey40", SES = "#D55E00",
                                 Holt = "#0072B2", ARRES = "#009E73")) +
  ggtitle("Univariate Models over the Hold-out Period (overlaid)") +
  ylab("Price (USD)") + xlab(NULL) +
  theme_minimal()


# =============================================================================
# 6. ONE-STEP-AHEAD FORECAST SUMMARY
# =============================================================================

cat("\n============================================================\n")
cat("ONE-STEP-AHEAD FORECAST - GOOGL (next trading day)\n")
cat("============================================================\n")

cat("SES    :", round(ses_onestep$mean,  2), "USD\n")
cat("Holt's :", round(holt_onestep$mean, 2), "USD\n")
cat("ARRES  :", round(arres_onestep,     2), "USD\n")
