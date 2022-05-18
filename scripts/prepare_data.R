library(tidyverse)
library(lubridate)
setwd("~/JpGU2022")
df <- read_csv("data/senjo/predicted/resnet.csv") %>%
  select(-X1)

cloudiness <- tibble(
  weather = c("clear", "moderate", "cloudy"),
  cloudiness = c(0, 1, 2)
)

df <- df %>%
  mutate(datetime = ymd_hm(datetime)) %>%
  filter(year(datetime)==2020) %>%
  arrange(datetime) %>%
  left_join(cloudiness, by="weather") %>%
  mutate(
    above_c = if_else(index < 8, cloudiness, 0),
    below_c = if_else(index >= 8, cloudiness, 0)
  ) %>%
  group_by(datetime) %>%
  summarise(
    above = sum(above_c) / 2,
    below = sum(below_c) / 3
  )


dt <- tibble(
  datetime = seq(ymd_hm("2020-01-01-0000"), ymd_hm("2020-12-31-2300"), by = "hour")
)

df <- dt %>%
  left_join(df)

write_csv(df, "data/senjo/predicted/2020.csv")

read_weather <- function(path) {
  read_csv(path) %>%
    slice(2:nrow(.))
}

weather <- map(list.files("data/senjo/weather/", full.names = T), read_weather) %>%
  reduce(left_join) %>%
  mutate(datetime = ymd_hms(datetime))

df <- df %>%
  left_join(weather)

df %>%
  write_csv("data/senjo/predicted/2020.csv")

mymean <- function(x) {
  mean(x, na.rm = T)
}

df %>%
  mutate(date = date(datetime)) %>%
  select(-datetime) %>%
  group_by(date) %>%
  summarise(across(.fns=mymean)) %>%
  rename(datetime=date) %>%
  write_csv("data/senjo/predicted/2020_daily.csv")

