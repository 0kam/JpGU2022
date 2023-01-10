library(tidyverse)

convert_label <- function (path) {
    read_csv(path) %>%
    mutate(
        label = case_when(
            label == "clear" ~ 0,
            label == "middle" ~ 1,
            label == "cloudy" ~ 2,
            label == "error" ~ 3
        )
    ) %>%
    write_csv(path)
}

labels = c("data/cyo/2016/labeled_CYO_2016_ground.csv", "data/cyo/2016/labeled_CYO_2016_sky.csv", "data/cyo/2020/labeled_CYO_2020_ground.csv", "data/cyo/2020/labeled_CYO_2020_sky.csv")
map(labels, convert_label)