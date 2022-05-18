library(tidyverse)
library(gtsummary)

df <- read_csv(
  "runs/efficientnet_b4_20220516_18_00_21/staritfied_cv.csv", # ground
  col_select = c(metrics, class, value, fold))

theme_gtsummary_mean_sd()
df %>%
  filter(metrics != "support") %>%
  filter(class != "accuracy") %>%
  mutate(class = ifelse(class == 3, "error", class)) %>%
  mutate(class = factor(class)) %T>%
  {
    group_by(., class, metrics) %>%
      summarise(
        mean = mean(value),
        sd = sd(value)
      ) %>%
      write_csv("table1.csv")
  } %>%
  pivot_wider(names_from = metrics, values_from = value) %>%
  select(-fold) %>%
  tbl_summary(
    by = c(class),
    label = list(
      precision ~ "Precision",
      recall ~ "Recall",
      `f1-score` ~ "F1 Score")
  ) %>%
  modify_header(label = "") %>%
  as_flex_table() %>%
  flextable::save_as_pptx(path = "cv_ground.pptx")

df <- read_csv(
  "runs/efficientnet_b4_20220517_12_32_28/staritfied_cv.csv", # ground
  col_select = c(metrics, class, value, fold))

theme_gtsummary_mean_sd()
df %>%
  filter(metrics != "support") %>%
  filter(class != "accuracy") %>%
  mutate(class = ifelse(class == 3, "error", class)) %>%
  mutate(class = factor(class)) %T>%
  {
    group_by(., class, metrics) %>%
      summarise(
        mean = mean(value),
        sd = sd(value)
      ) %>%
      write_csv("table1.csv")
  } %>%
  pivot_wider(names_from = metrics, values_from = value) %>%
  select(-fold) %>%
  tbl_summary(
    by = c(class),
    label = list(
      precision ~ "Precision",
      recall ~ "Recall",
      `f1-score` ~ "F1 Score")
  ) %>%
  modify_header(label = "") %>%
  as_flex_table() %>%
  flextable::save_as_pptx(path = "cv_sky.pptx")
