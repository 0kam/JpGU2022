library(tidyverse)
library(lubridate)
library(RColorBrewer)
library(animation)
setwd("JpGU2022/")

d <- "/media/okamoto/HDD4TB/JpGU2022/data/cyo"
paths <- c(
    str_c(d, "/2016/predicted_ground.csv"),
    str_c(d, "/2020/predicted_ground.csv")
)

# データ読込
load_data <- function(path) {
    read_csv(path,
    col_select = c(datetime, index, weather)) %>%
    mutate(datetime = ymd_hm(datetime)) %>%
    filter(weather != "error") %>%
    mutate(datetime = round_date(datetime, unit = "30 minutes")) %>%
    mutate(
        weather = as.integer(weather),
        year = year(datetime),
        month = month(datetime),
        day = day(datetime),
        hour = hour(datetime),
        minute = minute(datetime),
        x = index %% 4,
        y = index %/% 4
    ) %>%
    mutate(y = max(y) - y) # 左下が原点の座標系
}

df <- map_dfr(paths, load_data)
    

# 「雲海」の月ごと頻度
p1 <- df %>%
    mutate(
        ridge_weather = ifelse(y == 3, weather, NA),
        side_weather = ifelse(y != 3, weather, NA)
    ) %>% # 一番上の行を稜線と定義
    group_by(year, month, day, hour, minute) %>%
    summarise(
        ridge_weather = mean(ridge_weather, na.rm = T),
        side_weather = mean(side_weather, na.rm = T)
    ) %>%
    mutate(year = as.factor(year)) %>%
    #filter(minute == 0) %>% # 毎時30分のデータを排除
    mutate(unkai = (ridge_weather <= 0.25) & (side_weather >= 0.5)) %>%
    group_by(year, month, hour) %>%
    summarize(
      freq_unkai = mean(unkai, na.rm = T),
      supports = n()
    ) %>%
    mutate(freq_unkai = replace_na(freq_unkai, 0)) %>%
    filter(supports > 1) %>% # サンプルサイズが1の時間帯は省く
    mutate(month_name = str_c(as.character(month), "月")) %>%
    rowwise() %>%
    mutate(month = factor(month, labels = month_name)) %>%
    ggplot(aes(hour, freq_unkai, fill = year)) +
    geom_col(position = "dodge") +
    facet_wrap(~month) +
    labs(
      title = "雲海出現頻度の時間変化（月ごと）",
      x = "時刻", 
      y = "雲海が観測された頻度",
      caption = "格子の一番上の行に雲が少なく、
                 それ以外の格子が曇っている状態を「雲海あり」と判定"
    ) + 
    theme(plot.title = element_text(hjust = 0.5)) +
    # scale_y_continuous(breaks=seq(0,10,2)) +
    scale_x_continuous(breaks=seq(5,16,2))

ggsave("results/AJG2020Spr/unkai.png", p1)
#ggsave("results/AJG2020Spr/unkai_oclock.png", p1)

#  「稜線のみ雲」の月ごと頻度
p2 <- df %>%
  mutate(
    ridge_weather = ifelse(y == 3, weather, NA),
    side_weather = ifelse(y != 3, weather, NA)
  ) %>% # 一番上の行を稜線と定義
  group_by(year, month, day, hour, minute) %>%
  summarise(
    ridge_weather = mean(ridge_weather, na.rm = T),
    side_weather = mean(side_weather, na.rm = T)
  ) %>%
  #filter(minute == 0) %>% # 毎時30分のデータを排除
  mutate(year = as.factor(year)) %>%
  mutate(ridge_only = (ridge_weather >= 0.25) & (side_weather <= 0)) %>%
  group_by(year, month, hour) %>%
  summarize(
    n = mean(ridge_only, na.rm = T),
    supports = n()
  ) %>%
  filter(supports > 1) %>% # サンプルサイズが1の時間帯は省く
  # mutate(n = replace_na(n, 0)) %>%
  mutate(month_name = str_c(as.character(month), "月")) %>%
  rowwise() %>%
  mutate(month = factor(month, labels = month_name)) %>%
  ggplot(aes(hour, n, fill = year)) +
  geom_col(position = "dodge") +
  facet_wrap(~month) +
  labs(
    title = "稜線のみ雲がかかる頻度の時間変化（月ごと）",
    x = "時刻", 
    y = "稜線のみ雲がかかった頻度",
    caption = "格子の一番上の行に雲が少しでもあり、
                 それ以外の格子に全く雲がない状態をカウント"
  ) + 
  theme(plot.title = element_text(hjust = 0.5)) +
  # scale_y_continuous(breaks=seq(0,100,5), minor_breaks = seq(0, 100, 1)) +
  scale_x_continuous(breaks=seq(5,16,2))

p2
ggsave("results/AJG2020Spr/ridge.png", p2)
ggsave("results/AJG2020Spr/ridge_oclock.png", p2)

#  「カメラがガスに包まれる頻度」
p3 <- df %>%
  filter(!((x == 3) & (y == 0))) %>% # 右下の格子を除外
  group_by(year, month, day, hour, minute) %>%
  summarise(
    weather = mean(weather)
  ) %>%
  # filter(minute == 0) %>%  # 毎時30分のデータを排除
  mutate(year = as.factor(year)) %>%
  mutate(fog = (weather == 2)) %>%
  group_by(year, month, hour) %>%
  summarize(
    n = mean(fog),
    supports = n()
  ) %>%
  filter(supports > 1) %>%
  mutate(month_name = str_c(as.character(month), "月")) %>%
  rowwise() %>%
  mutate(month = factor(month, labels = month_name)) %>%
  ggplot(aes(hour, n, fill = year)) +
  geom_col(position = "dodge") +
  facet_wrap(~month) +
  labs(
    title = "カメラが霧に覆われた頻度（月ごと）",
    x = "時刻", 
    y = "霧に覆われた頻度",
    caption = "右下の格子以外がすべてcloudyなときを霧と定義した"
  ) + 
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_x_continuous(breaks=seq(5,16,2))

ggsave("results/AJG2020Spr/fog.png", p3)
ggsave("results/AJG2020Spr/fog_oclock.png", p3)

# 雲の量と雲の位置(上下)をプロット
library(gganimate)

path <- df %>%
  filter(year == 2016) %>%
  mutate(
    y = y - 1.5,
    weather2 = ifelse(weather==0, NA, weather)
  ) %>%
  mutate(weighted = (y * weather2) / 3)  %>%
  group_by(datetime) %>%
  summarise(
    cloudiness = mean(weather, na.rm = T) / 2,
    position = mean(weighted, na.rm = T),
    n = row_number()
  ) %>%
  mutate(position = ifelse(is.nan(position), 0, position)) %>%
  filter(n == 16) %>%
  mutate(month = month(datetime)) %>%
  mutate(month_name = str_c(as.character(month), "月")) %>%
  rowwise() %>%
  mutate(month = factor(month, labels = month_name)) %>%
  ungroup() %>%
  mutate(
    time = hour(datetime) + minute(datetime) / 60,
    date = as.factor(date(datetime)),
    day = day(datetime)
  ) %>%
  mutate(day = as.factor(day))

anim <- path %>%
  #filter((day %% 5) == 0) %>%
  ggplot(aes(time, position, color = day, size = cloudiness), group = date) +
  geom_jitter(show.legend = FALSE, alpha = 0.3) +
  scale_size(range = c(1, 8)) +
  facet_wrap(~month) +
  gganimate::transition_time(time) +
  ease_aes("linear") +
  labs(title = "時刻: {frame_time}", x = "雲の割合", y = "雲の高さ（1が最上段に集中、-1が最下段に集中）")

anim_save("results/AJG2020Spr/trajectory.gif", 
          anim, height = 600, width = 900, units = "px")

anim <- path %>%
  #filter((day %% 5) == 0) %>%
  ggplot(aes(cloudiness, position, color = day), group = date) +
  geom_jitter(show.legend = FALSE, alpha = 0.3) +
  facet_wrap(~month) +
  gganimate::transition_time(time) +
  ease_aes("linear") +
  labs(title = "時刻: {frame_time}", x = "雲の割合", y = "雲の高さ（1が最上段に集中、-1が最下段に集中）")

anim_save("results/AJG2020Spr/trajectory2.gif", anim)

anim <- path %>%
  filter((as.integer(day) %% 3) == 0) %>%
  filter(month == "7月") %>%
  ggplot(aes(cloudiness, position, color = day)) +
  geom_point(show.legend = FALSE, alpha = 0.5, size = 4) +
  geom_path(alpha = 0.3, show.legend = FALSE) +
  theme_minimal() +
  gganimate::transition_reveal(time) +
  ease_aes("linear") +
  labs(title = "時刻: {frame_along}", x = "雲の割合", y = "雲の高さ（1が最上段に集中、-1が最下段に集中）")

anim_save("results/AJG2020Spr/trajectory_2016_7.mp4", anim, renderer = ffmpeg_renderer())

p4 <- path %>%
  ggplot(aes(cloudiness, position, color = time), group = date) +
  geom_point(show.legend = T, size = 2, alpha = 0.75) +
  facet_wrap(~month) +
  scale_colour_viridis_c(option = "magma")

p4 <- path %>%
  ggplot(aes(time, position, color = cloudiness), group = date) +
  geom_jitter(show.legend = T, alpha = 0.5) +
  facet_wrap(~month) +
  scale_colour_viridis_c(option = "magma") +
  scale_size(range = c(1, 4))

p4

ggsave("results/AJG2020Spr/trajectory.png", p4)

# 雲の量と雲の位置(上下)の軌跡をクラスタリングしてみる

num_clusters <- 7

flat_p <- function(p) {
  p %>% 
    select(datetime, date, time, cloudiness, position) %>%
    pivot_wider(id_cols = date, names_from = time, values_from = c(cloudiness, position))
}

paths <- path %>% 
  filter((6.5 <= time) & (time <= 16)) %>%
  group_by(date) %>%
  group_split %>%
  map_dfr(flat_p) %>%
  na.omit()

res <- paths %>%
  select(-date) %>%
  kmeans(num_clusters)

paths$cluster <- res$cluster

path2 <- paths %>%
  select(date, cluster) %>%
  inner_join(path, by = "date") %>%
  mutate(cluster = as.factor(cluster))

p5 <- path2 %>%
  ggplot(aes(cloudiness, position, color = cluster), group = date) +
  geom_point(show.legend = T, size = 2, alpha = 0.75) +
  facet_wrap(~month)

ggsave("results/AJG2020Spr/kmeans.png", p5)

centers <- res$centers %>%
  as_tibble() %>%
  mutate(cluster = as.factor(1:num_clusters)) %>%
  pivot_longer(-cluster, names_to = c("var", "time"), names_sep = "_") %>%
  pivot_wider(names_from = var, values_from = value) %>%
  mutate(time = as.numeric(time)) %>%
  arrange(time)

p6 <- centers %>%
  ggplot(aes(cloudiness, position, color = cluster, alpha = time)) +
  geom_path()

ggsave("results/AJG2020Spr/kmeans_centers.png", p6)

anim <- centers %>%
  ggplot(aes(cloudiness, position, color = cluster)) +
  geom_point(show.legend = TRUE, alpha = 0.7, size = 4) +
  geom_path(alpha = 0.3, show.legend = FALSE) +
  theme_minimal() +
  gganimate::transition_reveal(time) +
  ease_aes("linear") +
  labs(title = "時刻: {frame_along}", x = "雲の割合", y = "雲の高さ（1が最上段に集中、-1が最下段に集中）")

anim_save("results/AJG2020Spr/trajectory_kmeans2.mp4", anim, renderer = ffmpeg_renderer())

anim <- centers %>%
  ggplot(aes(time, position, color = cluster, size = cloudiness), group = cluster) +
  geom_point(show.legend = FALSE) +
  scale_size(range = c(2, 12)) +
  gganimate::transition_time(time) +
  ease_aes("linear") +
  labs(title = "時刻: {frame_time}", x = "雲の割合", y = "雲の高さ（1が最上段に集中、-1が最下段に集中）")


anim_save("results/AJG2020Spr/trajectory_kmeans.gif", anim)

## 教師データの分布
library(tidyverse)
library(lubridate)
d16 <- read_csv("/media/okamoto/HDD4TB/JpGU2022/data/cyo/2016/labeled_CYO_2016_sky.csv") %>%
  mutate(year = as.factor(2016))

d20 <- read_csv("/media/okamoto/HDD4TB/JpGU2022/data/cyo/2020/labeled_CYO_2020_sky.csv") %>%
  mutate(year = as.factor(2020))

df <- bind_rows(d16, d20) %>%
  mutate(label = as.factor(label))

p <- df %>%
  ggplot(aes(x = label, fill = year)) +
  geom_histogram(stat = "count", position = "dodge") +
  labs(title = "教師データの分布")

ggsave("./results/AJG2020Spr/teacher_distribution.png", p)
