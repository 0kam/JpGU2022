library(rEDM)
library(tidyverse)

# 読み込みと標準化
df <- read_csv("data/senjo/predicted/2020_daily.csv")

# とりあえずプロット
p <- df %>%
  select(datetime, above, below) %>%
  slice(1:200) %>%
  pivot_longer(-datetime, "window", values_to="cloudiness") %>%
  ggplot(aes(x=datetime, y=cloudiness)) +
  geom_line(aes(color=window))
ggsave("cloudiness_daily.png",p, width=10, height=6)

df <- df %>%
  mutate(across(-datetime, scale)) %>%
  mutate(across(-datetime, as.vector))

block_above <- make_block(
  df$above, #元となる時系列データ
  max_lag=2 #時間遅れ軸数（t=0を含む）
)

p <- plot(block_above[,2], block_above[,3], type="o", pch=18,
     xlab="Above cloudiness (t)", ylab="Above cloudiness (t-1)", lwd=0.5)

block_below <- make_block(
  df$below, #元となる時系列データ
  max_lag=2 #時間遅れ軸数（t=0を含む）
)

p <- plot(block_below[,2], block_below[,3], type="o", pch=18,
     xlab="Below cloudiness (t)", ylab="Below cloudiness (t-1)", lwd=0.5)

#最適埋め込み次元の探索
simp_above <- simplex(df$above, 
                    E=1:20, #埋め込み次元を1~10まで試す
                    lib=c(1, nrow(df)), pred=c(1,nrow(df)), 
                    stats_only=TRUE, #予測値は出力せず、統計量だけ出力する
                    silent=TRUE)

par(mfrow=c(1, 3), las=1)
plot(rho~E, simp_above, type="o", pch=16)
plot(mae~E, simp_above, type="o", pch=16)
plot(rmse~E, simp_above, type="o", pch=16)

simp_below <- simplex(df$below, 
                      E=1:20, #埋め込み次元を1~10まで試す
                      lib=c(1, nrow(df)), pred=c(1, nrow(df)), 
                      stats_only=TRUE, #予測値は出力せず、統計量だけ出力する
                      silent=TRUE)

par(mfrow=c(1, 3), las=1)
plot(rho~E, simp_below, type="o", pch=16)
plot(mae~E, simp_below, type="o", pch=16)
plot(rmse~E, simp_below, type="o", pch=16)

