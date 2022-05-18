library(tidyverse)
library(lubridate)
library(RColorBrewer)
library(animation)

df <- read_csv("/home/okamoto/JpGU2022/data/cyo/2019_pred_ground.csv",
    col_select = c(datetime, index, weather)) %>%
    mutate(datetime = ymd_hm(datetime)) %>%
    filter(weather != "error") %>%
    mutate(
        weather = as.integer(weather),
        month = month(datetime),
        day = day(datetime),
        hour = hour(datetime),
        minute = minute(datetime),
        x = index %% 4,
        y = index %/% 4
    ) %>%
    mutate(y = max(y) - y)

# 通年通した時間変化
freq <- df %>%
    group_by(hour, minute, x, y) %>%
    summarise(cloud_freq = mean(weather) / 2) %>%
    ungroup()

img <- jpeg::readJPEG("data/cyo/ground.jpg")

plot_heat <- function(df) {
    time <- df %>%
    distinct(hour, minute)
    p <- df %>%
    ggplot(mapping = aes(x = x, y = y, fill = cloud_freq)) +
    annotation_raster(img, -0.5, 3.5, -0.5, 2.5) +
    geom_tile(alpha = 0.74) +
    scale_fill_viridis_c("Cloud Frequency",
     option = "magma", limits = c(0, 0.8))
    if (nrow(time) == 1) {
        time <- str_c(
            str_pad(time[1], 2, pad = 0), ":", str_pad(time[2], 2, pad = 0)
            )
        print(time)
        p <- p + ggtitle(time) +
            theme(
                plot.title = element_text(hjust = 0.5),
                text = element_text(size = 20)
                )
    }
    return(p)
}

plots <- freq %>%
    group_split(hour, minute) %>%
    map(plot_heat)

saveGIF({
    for (i in 1:length(plots)) {
        plot(plots[[i]])
    }
}, interval = 0.5, movie.name = "all.gif",
ani.width = 600, ani.height = 300
)

# 季節ごとの時間変化
as_season <- function(month) {
    case_when(
        month %in% c(12, 1, 2) ~ "winter",
        month %in% c(3, 4, 5) ~ "spring",
        month %in% c(6, 7, 8) ~ "summer",
        month %in% c(9, 10, 11) ~ "fall",
    )
}

freq <- df %>%
    mutate(season = as_season(month)) %>%
    group_by(season, hour, minute, x, y) %>%
    summarise(cloud_freq = mean(weather) / 2) %>%
    ungroup() %>%
    group_split(season)

for (d in freq) {
    season <- unique(d$season)
    plots <- d %>%
        group_split(hour, minute) %>%
        map(plot_heat)

    saveGIF({
        for (i in 1:length(plots)) {
            plot(plots[[i]])
        }
    }, interval = 0.5, movie.name = str_c(season, ".gif"),
    ani.width = 600, ani.height = 300
    )
}

# 月ごとの時間変化
freq <- df %>%
    group_by(month, hour, minute, x, y) %>%
    summarise(cloud_freq = mean(weather) / 2) %>%
    ungroup() %>%
    group_split(month)

for (d in freq) {
    month <- unique(d$month)
    plots <- d %>%
        group_split(hour, minute) %>%
        map(plot_heat)

    saveGIF({
        for (i in 1:length(plots)) {
            plot(plots[[i]])
        }
    }, interval = 0.5, movie.name = str_c(month, ".gif"),
    ani.width = 600, ani.height = 300
    )
}


# 時間ごとに、雲量の月変化を出す
plot_heat_monthly <- function(df) {
    month <- df %>%
        distinct(month)
    p <- df %>%
        ggplot(mapping = aes(x = x, y = y, fill = cloud_freq)) +
        annotation_raster(img, -0.5, 3.5, -0.5, 2.5) +
        geom_tile(alpha = 0.6) +
        scale_fill_viridis_c("Cloud Frequency",
            option = "magma", limits = c(0, 0.8))
    if (nrow(month) == 1) {
        month <- str_c(month, "月")
        print(month)
        p <- p + ggtitle(month) +
            theme(
                plot.title = element_text(hjust = 0.5),
                text = element_text(size = 20)
                )
    }
    return(p)
}

freq <- df %>%
    group_by(month, hour, x, y) %>%
    summarise(cloud_freq = mean(weather) / 2) %>%
    ungroup() %>%
    group_split(hour)

for (d in freq) {
    hour <- unique(d$hour)
    plots <- d %>%
        group_split(month) %>%
        map(plot_heat_monthly)

    saveGIF({
        for (i in 1:length(plots)) {
            plot(plots[[i]])
        }
    }, interval = 0.5, movie.name = str_c(hour, "oclock.gif"),
    ani.width = 600, ani.height = 300
    )
}

