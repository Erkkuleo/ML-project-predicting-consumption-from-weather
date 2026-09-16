# ============================================================
#  Pearson correlation matrix of features and label
#  Run after merge_electricity_weather.R
# ============================================================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

dataset <- read_csv("electricity_weather_merged.csv", show_col_types = FALSE)

# the label first, then the seven features
vars <- c("consumption_mwh", "temp_c", "wind_ms", "humid_pc",
          "hour_sin", "hour_cos", "weekday", "month")

cm <- dataset %>%
  select(all_of(vars)) %>%
  cor(method = "pearson", use = "complete.obs")

cat("\nPearson correlation matrix:\n")
print(round(cm, 3))


# ---- heatmap ----------------------------------------------
# keep the variables in the order above rather than alphabetical
cm_long <- as.data.frame(cm) %>%
  mutate(var1 = rownames(cm)) %>%
  pivot_longer(-var1, names_to = "var2", values_to = "r") %>%
  mutate(
    var1 = factor(var1, levels = vars),
    var2 = factor(var2, levels = rev(vars))
  )

p <- ggplot(cm_long, aes(var1, var2, fill = r)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", r)), size = 3) +
  scale_fill_gradient2(
    low = "#2166ac", mid = "white", high = "#b2182b",
    midpoint = 0, limits = c(-1, 1), name = "Pearson r"
  ) +
  labs(x = NULL, y = NULL,
       title = "Pearson correlation between features and label") +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid  = element_blank()
  ) +
  coord_fixed()

ggsave("correlation_matrix.png", p, width = 7, height = 6, dpi = 200)
cat("\nSaved correlation_matrix.png\n")


# ---- just the label column, sorted, for the report ---------
cat("\nCorrelation with consumption_mwh, strongest first:\n")
cm[, "consumption_mwh"] %>%
  .[names(.) != "consumption_mwh"] %>%
  sort(decreasing = TRUE) %>%
  round(3) %>%
  print()

