library(tidyverse)
# Class 1 (少し雲あり) と Class 2 (中程度雲あり)

convert_classes <- function(path) {
    path %>%
    read_csv() %>%
    mutate(
        label = if_else(label >= 2, label - 1, label)
        ) %>%
    write_csv(str_replace(path, ".csv", "_cat3.csv"))
}

c("data/cyo/2019_ground.csv", "data/cyo/2019_sky.csv") %>%
map(convert_classes)
