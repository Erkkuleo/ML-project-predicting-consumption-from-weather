# ============================================================
#  Chronological train / validation / test split
#  Run after merge_electricity_weather.R
#  Prints the set sizes needed for the Methods section.
# ============================================================

library(readr)
library(dplyr)
library(lubridate)

dataset <- read_csv("electricity_weather_merged.csv", show_col_types = FALSE) %>%
  arrange(time_utc)

# ---- split on whole calendar years (local time) -------------
# 2020-2023 -> training, 2024 -> validation, 2025 -> test
dataset <- dataset %>% mutate(year = year(local_time))

train <- dataset %>% filter(year %in% 2020:2023)
val   <- dataset %>% filter(year == 2024)
test  <- dataset %>% filter(year == 2025)

n_total <- nrow(dataset)

report <- tibble(
  set     = c("Training", "Validation", "Test", "Total"),
  period  = c("2020-2023", "2024", "2025", "2020-2025"),
  n       = c(nrow(train), nrow(val), nrow(test), n_total),
  share   = sprintf("%.1f%%", 100 * c(nrow(train), nrow(val),
                                      nrow(test), n_total) / n_total)
)

cat("\n=== Split sizes (paste these into the report) ===\n")
print(report, n = Inf)

# ---- checks ------------------------------------------------
cat("\nRows accounted for:", nrow(train) + nrow(val) + nrow(test),
    "of", n_total,
    if (nrow(train) + nrow(val) + nrow(test) == n_total) "  OK" else "  MISMATCH", "\n")

cat("\nDate ranges:\n")
cat("  train:", format(min(train$local_time)), "->", format(max(train$local_time)), "\n")
cat("  val  :", format(min(val$local_time)),   "->", format(max(val$local_time)),   "\n")
cat("  test :", format(min(test$local_time)),  "->", format(max(test$local_time)),  "\n")

# no overlap between sets?
cat("\nNo time overlap:",
    max(train$time_utc) < min(val$time_utc) &&
      max(val$time_utc)   < min(test$time_utc), "\n")

# hours per year, to spot gaps (a full year is 8760, leap year 8784)
cat("\nHours per year:\n")
dataset %>% count(year) %>% print(n = Inf)

# label distribution per set - they should look broadly similar
cat("\nConsumption (MWh) by set:\n")
bind_rows(
  train %>% mutate(set = "train"),
  val   %>% mutate(set = "val"),
  test  %>% mutate(set = "test")
) %>%
  group_by(set) %>%
  summarise(
    n    = n(),
    mean = round(mean(consumption_mwh)),
    sd   = round(sd(consumption_mwh)),
    min  = round(min(consumption_mwh)),
    max  = round(max(consumption_mwh)),
    .groups = "drop"
  ) %>%
  print()

# ---- save the splits ---------------------------------------
write_csv(select(train, -year), "train.csv")
write_csv(select(val,   -year), "val.csv")
write_csv(select(test,  -year), "test.csv")
cat("\nSaved train.csv, val.csv, test.csv\n")