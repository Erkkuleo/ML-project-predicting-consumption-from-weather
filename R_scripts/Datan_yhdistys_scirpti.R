# ============================================================
#  Merge Fingrid electricity consumption with FMI weather
#  Output: one row per hour, ready for regression models
# ============================================================
#
#  Before running:
#    1. Put the two CSV files in the same folder as this script.
#    2. Set the working directory to that folder
#       (RStudio: Session -> Set Working Directory -> To Source File Location)
#    3. Run STEP 1 first and look at the printed column names.
#       If the auto-detection in STEP 2 fails, set the names by hand
#       where the comments tell you to.
# ------------------------------------------------------------

library(readr)
library(dplyr)
library(lubridate)

# ---- file names: change these to match your downloads ----
fingrid_file <- "124_2020-01-01T0000_2025-12-31T2359.csv"
fmi_file     <- "Espoo Tapiola_ 1.1.2020 - 31.12.2025_6fb25248-0319-4b4a-9f8a-b6d4684ab422.csv"


# ============================================================
# STEP 1 - read the raw files and look at them
# ============================================================

# Fingrid exports are semicolon-separated with "." as decimal mark
elec_raw    <- read_delim(fingrid_file, delim = ";", show_col_types = FALSE)
# FMI marks missing observations with "-"
weather_raw <- read_csv(fmi_file,     show_col_types = FALSE,
                        na = c("", "NA", "-"))

cat("\n--- Fingrid columns ---\n"); print(names(elec_raw))
cat("\n--- FMI columns ---\n");     print(names(weather_raw))
cat("\n--- first rows of Fingrid ---\n"); print(head(elec_raw, 3))
cat("\n--- first rows of FMI ---\n");     print(head(weather_raw, 3))


# helper: find the first column whose name matches one of the patterns
pick <- function(df, patterns, what) {
  hits <- names(df)[grepl(paste(patterns, collapse = "|"),
                          names(df), ignore.case = TRUE)]
  if (length(hits) == 0)
    stop("Could not find a column for '", what,
         "'. Set it by hand. Available: ", paste(names(df), collapse = ", "))
  hits[1]
}


# ============================================================
# STEP 2 - tidy the electricity data
# ============================================================
# Fingrid gives a timestamp and a value. Newer exports use
# "startTime"/"value", older ones "Start time UTC"/a long description.
# If auto-detection fails, replace the two lines below with e.g.
#   col_time  <- "startTime"
#   col_value <- "value"

col_time  <- pick(elec_raw, c("^startTime$", "start.?time", "^time$"), "timestamp")
col_value <- pick(elec_raw, c("^value$", "consumption"), "consumption value")

cat("\nUsing electricity columns:", col_time, "/", col_value, "\n")

elec <- elec_raw %>%
  transmute(
    # read_delim usually parses the time already; ymd_hms on a POSIXct
    # would drop midnight rows (they print without "00:00:00")
    time_utc        = if (inherits(.data[[col_time]], "POSIXct"))
                        with_tz(.data[[col_time]], "UTC")
                      else ymd_hms(.data[[col_time]], tz = "UTC", quiet = TRUE),
    consumption_mwh = as.numeric(.data[[col_value]])
  ) %>%
  filter(!is.na(time_utc), !is.na(consumption_mwh)) %>%
  # a few rows start off the normal grid (e.g. 08:55 or 12:30:05) and
  # would be averaged into the wrong hour, so keep only :00/:15/:30/:45
  filter(second(time_utc) == 0, minute(time_utc) %% 15 == 0) %>%
  # col_time is the START of each interval (startTime), so flooring to the
  # hour puts every point into the hour it measured: 14:00, 14:15, 14:30
  # and 14:45 all belong to 14:00-15:00.
  # This also collapses the quarter-hourly data (from 13.6.2023 onwards)
  # and the older hourly data onto the same hourly grid.
  mutate(time_utc = floor_date(time_utc, "hour")) %>%
  group_by(time_utc) %>%
  summarise(
    consumption_mwh = mean(consumption_mwh),
    n_points        = n(),          # 1 for old hourly data, 4 for quarter-hourly
    .groups = "drop"
  )

# sanity check: how many raw points went into each hour?
cat("\nRaw points per hour (1 = hourly data, 4 = quarter-hourly):\n")
print(table(elec$n_points))

elec <- select(elec, -n_points)


# ============================================================
# STEP 3 - tidy the weather data
# ============================================================
# FMI exports usually have separate Year / Month / Day / Time columns.
# Some exports instead have a single datetime column.

has_ymd <- all(c("Year", "Month", "Day") %in% names(weather_raw))

if (has_ymd) {
  col_clock <- pick(weather_raw, c("^Time"), "time of day")
  weather <- weather_raw %>%
    mutate(
      time_utc = make_datetime(
        year  = as.integer(Year),
        month = as.integer(Month),
        day   = as.integer(Day),
        hour  = as.integer(substr(as.character(.data[[col_clock]]), 1, 2)),
        # FMI column is "Time [Local time]"; use "UTC" if you downloaded UTC
        tz    = if (grepl("local", col_clock, ignore.case = TRUE))
                  "Europe/Helsinki" else "UTC"
      ),
      time_utc = with_tz(time_utc, "UTC"),
      # when clocks go back in October, local 03:00 occurs twice. lubridate
      # maps both rows to the later (winter time) instant, so move the
      # first one back an hour to its real summer-time instant.
      time_utc = if_else(duplicated(time_utc, fromLast = TRUE),
                         time_utc - hours(1), time_utc)
    )
} else {
  col_dt  <- pick(weather_raw, c("time", "date"), "timestamp")
  weather <- weather_raw %>%
    mutate(time_utc = ymd_hms(.data[[col_dt]], tz = "UTC", quiet = TRUE))
}

# the weather variables you selected when downloading.
# Add or remove lines here to match your file.
col_temp  <- pick(weather, c("temperature"),        "air temperature")
col_wind  <- pick(weather, c("wind speed"),         "wind speed")
col_humid <- pick(weather, c("humidity"),           "relative humidity")

cat("\nUsing weather columns:", col_temp, "/", col_wind, "/", col_humid, "\n")

weather <- weather %>%
  transmute(
    time_utc = floor_date(time_utc, "hour"),
    temp_c   = as.numeric(.data[[col_temp]]),
    wind_ms  = as.numeric(.data[[col_wind]]),
    humid_pc = as.numeric(.data[[col_humid]])
  ) %>%
  filter(!is.na(time_utc)) %>%
  group_by(time_utc) %>%
  summarise(across(everything(), ~ mean(.x, na.rm = TRUE)), .groups = "drop")


# ============================================================
# STEP 4 - join and build the features
# ============================================================

dataset <- inner_join(elec, weather, by = "time_utc") %>%
  mutate(
    local_time = with_tz(time_utc, "Europe/Helsinki"),
    hour       = hour(local_time),                 # 0-23
    weekday    = wday(local_time, week_start = 1), # 1 = Monday
    is_weekend = as.integer(weekday >= 6),
    month      = month(local_time),
    # cyclic encoding: hour 23 and hour 0 should be close together
    hour_sin   = sin(2 * pi * hour / 24),
    hour_cos   = cos(2 * pi * hour / 24)
  ) %>%
  select(time_utc, local_time, consumption_mwh,
         temp_c, wind_ms, humid_pc,
         hour, hour_sin, hour_cos, weekday, is_weekend, month) %>%
  arrange(time_utc)

# drop rows with missing values (report how many you lose)
n_before <- nrow(dataset)
dataset  <- dataset %>% filter(if_all(everything(), ~ !is.na(.x)))
cat("\nDropped", n_before - nrow(dataset), "rows with missing values\n")


# ============================================================
# STEP 5 - check it, then save
# ============================================================

cat("\nFinal dataset:", nrow(dataset), "rows,", ncol(dataset), "columns\n")
cat("Date range:", format(min(dataset$local_time)), "to",
    format(max(dataset$local_time)), "\n\n")
print(summary(dataset))

# write_csv would convert local_time back to UTC, so store it as text
dataset %>%
  mutate(local_time = format(local_time, "%Y-%m-%d %H:%M:%S")) %>%
  write_csv("electricity_weather_merged.csv")
cat("\nSaved to electricity_weather_merged.csv\n")

# quick sanity plot: consumption should fall as temperature rises
plot(dataset$temp_c, dataset$consumption_mwh,
     pch = ".", col = rgb(0, 0, 0, 0.2),
     xlab = "Air temperature (C)",
     ylab = "Electricity consumption (MWh/h)",
     main = "Consumption vs temperature")