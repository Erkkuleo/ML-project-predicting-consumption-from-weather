# ============================================================
#  Exploratory plots: consumption vs each feature
#  Run after merge_electricity_weather.R
# ============================================================

library(readr)
library(dplyr)
library(ggplot2)

dataset <- read_csv("electricity_weather_merged.csv", show_col_types = FALSE)

theme_set(theme_minimal(base_size = 9))

ylab_txt <- "Consumption (MWh)"

# ---- 1. temperature (the key relationship) ----
p1 <- ggplot(dataset, aes(temp_c, consumption_mwh)) +
  geom_hex(bins = 50) +                       # 52k points: hexbin beats scatter
  scale_fill_viridis_c(trans = "log10", guide = "none") +
  geom_smooth(method = "gam", colour = "red", linewidth = 0.6, se = FALSE) +
  labs(x = "Air temperature (C)", y = ylab_txt,
       title = "Consumption vs temperature")

# ---- 2. hour of day ----
p2 <- dataset %>%
  mutate(hour = round(atan2(hour_sin, hour_cos) / (2 * pi) * 24) %% 24) %>%
  ggplot(aes(factor(hour), consumption_mwh)) +
  geom_boxplot(outlier.size = 0.2, fill = "steelblue", alpha = 0.6) +
  scale_x_discrete(breaks = seq(0, 23, 3)) +
  labs(x = "Hour of day (local)", y = ylab_txt,
       title = "Consumption by hour of day")

# ---- 3. weekday ----
p3 <- ggplot(dataset, aes(factor(weekday), consumption_mwh)) +
  geom_boxplot(outlier.size = 0.2, fill = "darkorange", alpha = 0.6) +
  scale_x_discrete(labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")) +
  labs(x = "Weekday", y = ylab_txt,
       title = "Consumption by weekday")

# ---- 4. month ----
p4 <- ggplot(dataset, aes(factor(month), consumption_mwh)) +
  geom_boxplot(outlier.size = 0.2, fill = "forestgreen", alpha = 0.6) +
  labs(x = "Month", y = ylab_txt,
       title = "Consumption by month")

# ---- 5. wind speed ----
p5 <- ggplot(dataset, aes(wind_ms, consumption_mwh)) +
  geom_hex(bins = 40) +
  scale_fill_viridis_c(trans = "log10", guide = "none") +
  geom_smooth(method = "gam", colour = "red", linewidth = 0.6, se = FALSE) +
  labs(x = "Wind speed (m/s)", y = ylab_txt,
       title = "Consumption vs wind speed")

# ---- 6. relative humidity ----
p6 <- ggplot(dataset, aes(humid_pc, consumption_mwh)) +
  geom_hex(bins = 40) +
  scale_fill_viridis_c(trans = "log10", guide = "none") +
  geom_smooth(method = "gam", colour = "red", linewidth = 0.6, se = FALSE) +
  labs(x = "Relative humidity (%)", y = ylab_txt,
       title = "Consumption vs humidity")

# ---- save each plot as its own image in eda_plots/ ----
out_dir <- "eda_plots"
dir.create(out_dir, showWarnings = FALSE)

plots <- list(
  "01_consumption_vs_temperature" = p1,
  "02_consumption_by_hour"        = p2,
  "03_consumption_by_weekday"     = p3,
  "04_consumption_by_month"       = p4,
  "05_consumption_vs_wind_speed"  = p5,
  "06_consumption_vs_humidity"    = p6
)

for (name in names(plots)) {
  file <- file.path(out_dir, paste0(name, ".png"))
  ggsave(file, plots[[name]], width = 5, height = 3.5, dpi = 200, bg = "white")
  cat("Saved", file, "\n")
}

# ---- correlations between the numeric features and the label ----
cat("\nCorrelation with consumption:\n")
dataset %>%
  select(consumption_mwh, temp_c, wind_ms, humid_pc) %>%
  cor() %>%
  round(3) %>%
  print()

