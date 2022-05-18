library(tidyverse)
library(rEDM)
library(rgl)
library(furrr)

df <- read_csv("data/senjo/predicted/2020_daily.csv") %>%
  mutate(across(-datetime, scale)) %>%
  mutate(across(-datetime, as.vector)) %>%
  mutate(across(.fns = function(v) {ifelse(is.nan(v), NA, v)})) %>%
  fill(temperature, daylight_hours, .direction = "down")

df[is.nan(df)] <- NA

get_E <- function(vec) {
  simplex(
    vec,
    E = 1:20,
    stats_only = T
  ) %>%
    arrange(-rho) %>%
    slice(1) %>%
    pull(E)
}

Es <- df %>%
  select(-datetime) %>%
  map(get_E) %>%
  as_tibble()

set.seed(1)

# causality col1 <- col2
my_ccm <- function(d, tp) {
  cols <- colnames(d)
  E1 <- select(Es, cols[1]) %>% pull()
  E2 <- select(Es, cols[2]) %>% pull()
  ccm_res <- d %>%
    ccm(
      E = E1,
      tp=tp,
      lib_sizes = seq(E1+1,nrow(d),by=50)
    )
  res <- ccm_means(ccm_res)
  ccm_res_rev <- d %>%
    ccm(
      E = E2,
      tp = tp,
      lib_sizes = seq(E2+1,nrow(d),by=50),
      lib_column = 2,
      target_column = 1
    )
  res_rev <- ccm_means(ccm_res_rev)
  
  # Causality test with seasonal surrogat data
  surrogate <- d[,1] %>%
    make_surrogate_seasonal(
      T_period=366, #季節の周期
      num_surr=20) #作るデータの数
  
  ccm_sur <- function(s) {
    bind_cols(s, d[,2]) %>%
      ccm(
        E = E1,
        tp=tp,
        lib_sizes = seq(E1+1,nrow(d),by=50)
      ) %>%
      ccm_means()
  }
  
  s <- surrogate %>% 
    as_tibble()
  plan(multisession)
  sur_res <- future_map(s, ccm_sur) %>%
    reduce(rbind) %>%
    ccm_means(FUN=quantile, probs=0.95)
  
  # Reverse causality test with seasonal surrogate
  surrogate_rev <- d[,2] %>%
    make_surrogate_seasonal(
      T_period=366, #季節の周期
      num_surr=20) #作るデータの数
  
  ccm_sur_rev <- function(s) {
    bind_cols(s, d[,1]) %>%
      ccm(
        E = E2,
        tp=tp,
        lib_sizes = seq(E2+1,nrow(d),by=50),
      ) %>%
      ccm_means()
  }
  
  srr <- surrogate_rev %>% 
    as_tibble()
  plan(multisession)
  sur_res_rev  <- future_map(srr, ccm_sur_rev) %>%
    reduce(rbind) %>%
    ccm_means(FUN=quantile, probs=0.95)
  
  res_all <- list(
    res = res,
    res_rev = res_rev,
    sur_res = sur_res,
    sur_res_rev = sur_res_rev
  )
  
  ns <- names(res_all)
  cols <- colnames(d)
  add_trial <- function(n) {
    res_all[[n]] %>%
      mutate(
        from = if_else(str_detect(n, "rev"), cols[1], cols[2]),
        to = if_else(str_detect(n, "rev"), cols[2], cols[1]),
        surrogate = if_else(str_detect(n, "sur"), "surrogate", "actual"),
        tp = tp
        )  
  }
  res_all <- ns %>%
    map(add_trial) %>%
    reduce(rbind)
  
  return(res_all)
}

draw_ccm_result <- function(res) {
  title <- str_c("Causality from " , unique(res$from) , " to " , unique(res$to))
  ggplot(res, aes(x=lib_size, y=rho)) +
    geom_line(aes(color=surrogate)) + 
    labs(title = title)
}

tps <- 1:3

res_ah <- map(tps, function(t) my_ccm(select(df, above, humidity), t)) %>%
  reduce(bind_rows)

res_bh <- map(tps, function(t) my_ccm(select(df, below, humidity), t)) %>%
  reduce(bind_rows)

res_ap <- map(tps, function(t) my_ccm(select(df, above, pressure), t)) %>%
  reduce(bind_rows)

res_bp <- map(tps, function(t) my_ccm(select(df, below, pressure), t)) %>%
  reduce(bind_rows)

res_at <- map(tps, function(t) my_ccm(select(df, above, temperature), t)) %>%
  reduce(bind_rows)

res_bt <- map(tps, function(t) my_ccm(select(df, below, temperature), t)) %>%
  reduce(bind_rows)

res_ad <- map(tps, function(t) my_ccm(select(df, above, daylight_hours), t)) %>%
  reduce(bind_rows)

res_bd <- map(tps, function(t) my_ccm(select(df, below, daylight_hours), t)) %>%
  reduce(bind_rows)

res_ab <- map(tps, function(t) my_ccm(select(df, above, below), t)) %>%
  reduce(bind_rows)

res_all <- list(
  res_ah,
  res_bh,
  res_ap,
  res_bp,
  res_at,
  res_bt,
  res_ad,
  res_bd,
  res_ab
)

res_all <- read_csv("ccm_result.csv")
weathers <- c("daylight_hours", "humidity", "pressure", "temperature")

plot_ccm <- function(weather) {
  p <- res_all %>%
    filter(from == weather) %>%
    ggplot(aes(x=lib_size, y=rho, color=surrogate)) +
    geom_line() +
    facet_grid(to~tp) +
    ggtitle(str_c("Causality test from ", weather, " to cloudiness")) +
    theme(plot.title=element_text(hjust = 0.5, size=18)) 
  ggsave(str_c("ccm_", weather, ".png"), p)
}

weathers %>%
  map(plot_ccm)

plot_ccm("below")
