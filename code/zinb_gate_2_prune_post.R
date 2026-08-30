
# ------------------------------------
# config.
# ------------------------------------

# If running in RStudio, set working dir to script location (optional)
if (requireNamespace("rstudioapi", quietly = TRUE) &&
  rstudioapi::isAvailable()) {
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  }

pacman::p_load(
  data.table, dplyr, tibble,
  keras, tensorflow, reticulate, pROC, ggplot2
)

# set virtual env.
use_virtualenv("~/v_environment/r-tf", required = TRUE)

# ---- Determinism toggles (set before loading tensorflow/keras) ----
Sys.setenv(PYTHONHASHSEED = "0")      # Python hash seed
Sys.setenv(TF_DETERMINISTIC_OPS = "1")# request deterministic TF ops (TF>=2.13)

set_global_seed <- function(seed = 2026) {
  set.seed(seed)

  if (reticulate::py_module_available("numpy")) {
    reticulate::import("numpy")$random$seed(as.integer(seed))
  }

  tensorflow::tf$random$set_seed(as.integer(seed))
}
set_global_seed(2026)

outdir <- file.path(getwd(), "post_results")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

#######################################################################################
# functions
#######################################################################################
source(file.path(getwd(), "functions_zinb_gate_2_prune.R"))


#######################################################################################
# check performance. trva and te
#######################################################################################

f <- list.files(file.path(getwd(), "trva_infer"), full.names = T)

trva_df <- fread(grep("trva_pred",f, value=T))
te_df <- fread(grep("te_pred",f, value=T))
tr_va_te <- dplyr::bind_rows(trva_df, te_df)

# -------- vote counts
perf_tbl <- tr_va_te %>%
  dplyr::group_by(set) %>%
  dplyr::summarise(
    auc = dplyr::first(auc),
    r2_corr = dplyr::first(r2_corr),
    mae = dplyr::first(mae),
    rmse = dplyr::first(rmse),
    mean_true = mean(true, na.rm = TRUE),
    mean_pred = mean(pred, na.rm = TRUE),
    .groups = "drop"
  )
print(perf_tbl)

# Plot: predicted vs observed
reg_stats <- tr_va_te %>%
  dplyr::group_by(set) %>%
  dplyr::summarise(
    beta = {
      den <- sum(true^2, na.rm = TRUE)
      if (den <= .Machine$double.eps) NA_real_ else sum(true * pred, na.rm = TRUE) / den
    },
    r2_reg = {
      den <- sum(true^2, na.rm = TRUE)
      if (den <= .Machine$double.eps || is.na(beta)) {
        NA_real_
      } else {
        yfit <- beta * true
        sse <- sum((pred - yfit)^2, na.rm = TRUE)
        sst0 <- sum(pred^2, na.rm = TRUE)
        if (sst0 <= .Machine$double.eps) NA_real_ else 1 - sse / sst0
      }
    },
    .groups = "drop"
  )

fit_lines <- tr_va_te %>%
  dplyr::group_by(set) %>%
  dplyr::summarise(
    min_true = min(true, na.rm = TRUE),
    max_true = max(true, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::left_join(reg_stats, by = "set") %>%
  dplyr::rowwise() %>%
  dplyr::mutate(
    true_grid = list(seq(min_true, max_true, length.out = 200)),
    fitted_grid = list(
      if (is.na(beta)) rep(NA_real_, 200) else pmax(0, beta * true_grid)
    )
  ) %>%
  dplyr::ungroup() %>%
  tidyr::unnest(c(true_grid, fitted_grid)) %>%
  dplyr::transmute(
    set,
    lp_true = log1p(true_grid),
    lp_fitted = log1p(fitted_grid)
  )

plot_df <- tr_va_te %>%
  dplyr::left_join(reg_stats, by = "set") %>%
  dplyr::mutate(
    lp_true = log1p(true),
    lp_pred = log1p(pred),
    pos.x = max(lp_true, na.rm = TRUE) * 0.68,
    pos.y = max(lp_pred, na.rm = TRUE) * 0.98,
    label = sprintf(
      "AUC = %.3f\nβ = %.3f\nR² = %.3f",
      auc, beta, r2_corr
    )
  ) %>%
  dplyr::ungroup()

ggplot(plot_df, aes(x = lp_true, y = lp_pred)) +
  facet_wrap(~ set, nrow = 1) +
  geom_point(color = "darkred", alpha = 0.75, size = 1, shape = 1) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.5) +
  geom_line(
    data = fit_lines,
    aes(x = lp_true, y = lp_fitted),
    inherit.aes = FALSE,
    color = "darkgreen",
    linewidth = 1,
    linetype = "dashed"
  ) +
  geom_text(
    aes(x = pos.x, y = pos.y, label = label),
    hjust = 1,
    vjust = 1,
    size = 3
  ) +
  labs(
    x = "log1p(true votes)",
    y = "log1p(predicted votes)",
  ) +
  theme_bw()

ggsave(
  filename = file.path(outdir, "accuracy_trva_test.png"),
  width = 10,
  height = 5
)


# -------- vote share
temp <- copy(tr_va_te)
temp[, sum_true:=sum(true), by = .(time, lans_kommun_kod, kommun_name)]
temp[, sum_pred:=sum(pred), by = .(time, lans_kommun_kod, kommun_name)]
temp[, true_share := true / sum_true]
temp[, pred_share := pred / sum_pred]
vote_share <- temp

# check sum
temp <- vote_share[
  , .(
    check_true=sum(true_share),
    check_pred=sum(pred_share)
  ),
  by = .(time, lans_kommun_kod, kommun_name)
]
summary(temp$check_true)
summary(temp$check_pred)

# overall
temp <- copy(vote_share)
temp[, share_err := pred_share - true_share]
temp[, abs_share_err := abs(share_err)]
vote_share_perf <- temp[, .(
    r2_share = cor(true_share, pred_share, use = "complete.obs")^2,
    mae_share = mean(abs_share_err, na.rm = TRUE),
    rmse_share = sqrt(mean(share_err^2, na.rm = TRUE)),
    mean_true_share = mean(true_share, na.rm = TRUE),
    mean_pred_share = mean(pred_share, na.rm = TRUE)
  ), by=set]
vote_share_perf 


# Plot: predicted vs observed
reg_stats <- vote_share %>%
  dplyr::group_by(set) %>%
  dplyr::summarise(
    beta = {
      den <- sum(true_share^2, na.rm = TRUE)
      if (den <= .Machine$double.eps) NA_real_ else sum(true_share * pred_share, na.rm = TRUE) / den
    },
    r2_reg = {
      den <- sum(true_share^2, na.rm = TRUE)
      if (den <= .Machine$double.eps || is.na(beta)) {
        NA_real_
      } else {
        yfit <- beta * true_share
        sse <- sum((pred_share - yfit)^2, na.rm = TRUE)
        sst0 <- sum(pred_share^2, na.rm = TRUE)
        if (sst0 <= .Machine$double.eps) NA_real_ else 1 - sse / sst0
      }
    },
    .groups = "drop"
  )

fit_lines <- vote_share %>%
  dplyr::group_by(set) %>%
  dplyr::summarise(
    min_true = min(true_share, na.rm = TRUE),
    max_true = max(true_share, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::left_join(reg_stats, by = "set") %>%
  dplyr::rowwise() %>%
  dplyr::mutate(
    true_grid = list(seq(min_true, max_true, length.out = 200)),
    fitted_grid = list(
      if (is.na(beta)) rep(NA_real_, 200) else pmax(0, beta * true_grid)
    )
  ) %>%
  dplyr::ungroup() %>%
  tidyr::unnest(c(true_grid, fitted_grid)) %>%
  dplyr::transmute(
    set,
    lp_true = (true_grid),
    lp_fitted = (fitted_grid)
  )

plot_df <- vote_share %>%
  dplyr::left_join(reg_stats, by = "set") %>%
  dplyr::mutate(
    lp_true = (true_share),
    lp_pred = (pred_share),
    pos.x = max(lp_true, na.rm = TRUE) * 0.68,
    pos.y = max(lp_pred, na.rm = TRUE) * 0.98,
    label = sprintf(
      "AUC = %.3f\nβ = %.3f\nR² = %.3f",
      auc, beta, r2_reg
    )
  ) %>%
  dplyr::ungroup()

ggplot(plot_df, aes(x = true_share, y = pred_share)) +
  facet_wrap(~ set, nrow = 1) +
  geom_point(color = "darkred", alpha = 0.75, size = 1, shape = 1) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.5) +
  geom_line(
    data = fit_lines,
    aes(x = lp_true, y = lp_fitted),
    inherit.aes = FALSE,
    color = "darkgreen",
    linewidth = 1,
    linetype = "dashed"
  ) +
  geom_text(
    aes(x = pos.x, y = pos.y, label = label),
    hjust = 1,
    vjust = 1,
    size = 3
  ) +
  labs(
    x = "true vote share",
    y = "predicted votes share",
  ) +
  theme_bw()

ggsave(
  filename = file.path(outdir, "accuracy_trva_test_share.png"),
  width = 10,
  height = 5
)

#######################################################################################
# explain by trva
#######################################################################################

# ------------------------ Analyzie embedding
loaded <- load_zinb_model(file.path(getwd(), "trva_mod"))
dt <- fread("preprocessed_data.csv")

# ----- helper
get_embedding_matrix <- function(model, layer_name = NULL, index = 1) {
  # If a layer name is provided, use it
  if (!is.null(layer_name)) {
    lyr <- model %>% get_layer(layer_name)
    return(lyr$get_weights()[[1]])
  }
  # Otherwise, locate embeddings by class
  emb_layers <- Filter(
    function(x) grepl("Embedding", class(x)[1], ignore.case=TRUE),
    model$layers
  )
  if (length(emb_layers) < index) stop("No embedding layer at given index.")
  emb_layers[[index]]$get_weights()[[1]]
}

# ---- party embedding variance
#  Extract party embedding matrix 
emb_party_mat <- get_embedding_matrix(loaded$model, index = 1)  
hist(emb_party_mat)

#  Variance per latent dimension 
party_dim_variance <- apply(emb_party_mat, 2, var)
party_dim_variance

# ---- Party Embedding Norms (salience / influence)
# Euclidean norm per party (>= importance of each party in latent space)
party_norms <- apply(emb_party_mat, 1, function(r) sqrt(sum(r^2)))

# Combine with party labels
party_norms_tbl <- tibble::tibble(
  party_factor = seq_along(party_norms),
  norm = party_norms
) %>%
  dplyr::left_join(dt[, .(party_factor, pty_short, pty_name)] %>% unique(), by="party_factor") |> 
  as.data.table()

party_norms_tbl[order(-norm)]
hist(party_norms_tbl$norm)

# ---- PCA on party embeddings 
pca_party <- prcomp(emb_party_mat, center = TRUE, scale. = T)

pca_df <- tibble::tibble(
  pc_x = pca_party$x[,1],
  pc_y = pca_party$x[,2],
  party_factor = seq_len(nrow(pca_party$x))
) %>%
  dplyr::left_join(dt[, .(party_factor, pty_name, pty_short)] %>% unique(), by="party_factor") |> 
  mutate(main=ifelse(pty_short %in% c("l","s","sd","m","c","v","mp","kd"), 1, 0))

ggplot(pca_df |> filter(main==0), aes(pc_x, pc_y)) +
  geom_point(color="grey80", size =1.5) +
  geom_hline(yintercept = 0, color="grey60") + geom_vline(xintercept = 0, color="grey60") +
  geom_point(data=pca_df |> filter(main==1), size = 3, color="steelblue") +
  ggrepel::geom_text_repel(
    data=pca_df |> filter(main==1), 
    aes(label = pty_name), max.overlaps = 10) +
  theme_minimal(base_size = 14) +
  labs(
    x = "PCA 1 (largest variance)",
    y = "PCA 2"
  ) 
ggsave(file.path(outdir, "emb_party_pca.png"), width=8, height = 8)


#######################################################################################
# forecast by trvate
#######################################################################################

f <- list.files(file.path(getwd(), "trvate_infer"), full.names = T)

forecast_df <- fread(grep("2026_party_kommun.csv",f, value=T))

# -------- national ranking 
tmp <- copy(forecast_df)

nat_elig_voter <- tmp[,.(kommun_name,tot_elig_voter)] |> distinct()
nat_elig_voter <- sum(nat_elig_voter$tot_elig_voter)

nat_rank <- tmp[
  pty_name != "",
  .(
    pred_votes = sum(pred) |> as.integer()
  ),
  by = .(pty_name)
]

nat_rank[
  ,
  `:=`(
    pred_per_voter  = round(pred_votes / nat_elig_voter, 4),
    pred_vote_share = round(pred_votes / sum(pred_votes), 4)
  )
]

setorder(nat_rank, -pred_vote_share)
nat_rank[, pred_vote_share:=paste(round(pred_vote_share*100,2), "%")]
nat_rank[, pred_per_voter:=paste(round(pred_per_voter*100,2), "%")]
nat_rank <- nat_rank[1:top_n]
nat_rank[, pred_per_voter:=NULL]
print(nat_rank)

library(knitr)
kable(
 nat_rank,
  format = "latex",
  booktabs = TRUE,
  caption = ""
)

sel_pty <- c(
  "arbetarepartiet-socialdemokraterna",
   "moderaterna",
   "sverigedemokraterna" ,
   "centerpartiet",
   "vänsterpartiet",
   "kristdemokraterna",
   "miljöpartiet de gröna",
   "liberalerna (tidigare folkpartiet)"
)

#-----------------------------------------
# map forecast vote share
#-----------------------------------------

# -------- vote share by kommun
temp <- copy(forecast_df)
temp[, sum_pred:=sum(pred), by = .(time,  kommun_name)]
temp[, pred_share := pred / sum_pred]
vote_share <- temp

# check sum
temp <- vote_share[
  , .(
    check_pred=sum(pred_share)
  ),
  by = .(time, kommun_name)
]
summary(temp$check_pred)

# -------- map pred share
library(sf)
temp <- vote_share[!pty_name %in% sel_pty]
temp <- temp[,.(pred_share=sum(pred_share)), by=c("lans_kommun_kod")]
temp[, pty_name:="other"]

sel <- grep("lans_kommun|pred_share|pty_name", names(vote_share), value = T)
map_df <- vote_share[ pty_name %in% sel_pty, ..sel]
map_df <- rbind(map_df,temp)

# read shapes
kommun_shp <- st_read(file.path(getwd(), "input_data/alla_kommuner.shp"))
kommun_shp$KOM <- as.numeric(kommun_shp$KOM)

# all municipalities from shapefile
all_kommun <- unique(kommun_shp$KOM)
# all parties in map_df
all_party <- unique(map_df$pty_name)

# complete grid
full_grid <- CJ(
  lans_kommun_kod = all_kommun,
  pty_name = all_party,
  unique = TRUE
)

# join predictions
map_df <- merge(
  full_grid,
  map_df,
  by = c("lans_kommun_kod", "pty_name"),
  all.x = TRUE
)

# replace missing predictions with zero
# map_df[is.na(pred_share), pred_share := 0]

# attach geometry
map_df <- merge(
  kommun_shp,
  map_df,
  by.x = "KOM",
  by.y = "lans_kommun_kod",
  all.x = TRUE
)

# Relevel pty_short by lvl
map_df <- map_df %>%
  ungroup() %>%
  mutate(
    pty_name = factor(
      pty_name,
      levels = c(sel_pty, "other")
    )
  )

ggplot(map_df ) + #|> filter(pty_name!="other")
  geom_sf(aes(fill = pred_share*100), color = NA) +
  facet_wrap(.~pty_name, ncol=3, strip.position = "left") +
  scale_fill_viridis_c(
    option = "viridis",
    direction = -1,        # <-- reverse color order
    na.value = "grey90",
    name = "Vote share \n in % "
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.major = element_blank(),   # remove major grid
    panel.grid.minor = element_blank(),   # remove minor grid
    panel.background = element_rect(fill = "#d5f3fe", color = NA),
    axis.text = element_blank(),          # optional: hide axis numbers
    axis.ticks = element_blank(),         # optional: hide axis ticks
    axis.title = element_blank(),          # optional: remove axis titles
    legend.position ="right"
  ) +
  coord_sf(expand = TRUE)
ggsave(file.path(outdir, "forecast_map.png"), width=8, height = 9)


# ---- map the party with the largest predicted vote share (pred_share) within each lans_kommun_kod
# rank parties within municipality
top3_party_kommun <- vote_share[
  pty_name != "",
  .SD[order(-pred_share)][1:min(.N, 3)],
  by = .(lans_kommun_kod, kommun_name)
]

# add rank (1 = largest, 2 = second, 3 = third)
top3_party_kommun[
  ,
  rank := seq_len(.N),
  by = .(lans_kommun_kod, kommun_name)
]

# collapse smaller parties, major parties + rank1 parties
rank1_pty <- c(top3_party_kommun[rank %in% c(1)]$pty_name |> unique(), sel_pty) |> unique()
top3_party_kommun[
  ,
  pty_name := fifelse(pty_name %in% rank1_pty, pty_name, "other")
]

# cleaner facet labels
top3_party_kommun[, rank_lab := factor(
  rank,
  levels = c(1, 2, 3),
  labels = c("Largest party", "Second largest", "Third largest")
)]

# merge with geometry
map_df <- merge(
  kommun_shp,
  top3_party_kommun,
  by.x = "KOM",
  by.y = "lans_kommun_kod",
  all.x = FALSE
)

major_cols <- c(
  "arbetarepartiet-socialdemokraterna" = "#E8112D",  # S
  "moderaterna"                        = "#006AB3",  # M
  "sverigedemokraterna"                = "#f5e751",  # SD
  "centerpartiet"                      = "#009933",  # C
  "vänsterpartiet"                     = "#a20b0b",  # V
  "kristdemokraterna"                  = "#c59702",  # KD
  "miljöpartiet de gröna"              = "#83CF39",  # MP
  "liberalerna (tidigare folkpartiet)" = "#6BB7E8",  # L
  "other"                              = "grey60"
)

# remaining parties
other_parties <- setdiff(
  unique(top3_party_kommun$pty_name),
  names(major_cols)
)

library(randomcoloR)
library(farver)

# Major-party colors
major_hex <- unname(major_cols)

# Generate many candidate colors
cand <- distinctColorPalette(300)

# Convert to Lab color space
major_lab <- decode_colour(major_hex, to = "lab")
cand_lab  <- decode_colour(cand, to = "lab")

# Distance from each candidate to nearest major color
min_dist <- apply(
  compare_colour(cand_lab, major_lab, from_space = "lab"),
  1,
  min
)

# Keep only colors sufficiently different from majors
cand_keep <- cand[min_dist > 50]  # increase threshold for more separation

extra_cols <- setNames(
  cand_keep[seq_along(other_parties)],
  other_parties
)

party_cols <- c(major_cols, extra_cols)

ggplot(map_df) +
  geom_sf(aes(fill = pty_name), color = NA) +
  facet_wrap(~rank_lab, nrow = 1) +
  scale_fill_manual(
    values = party_cols,
    na.value = "grey90",
    name = "Party"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank(),
    legend.position = "right",
    strip.text = element_text(size=13)
  ) +
  coord_sf(expand = TRUE)

ggsave(file.path(outdir, "forecast_map_win_party.png"), width=10, height = 6)



#-----------------------------------------
# map forecast 1-pi, prob receiving non-zero vote
#-----------------------------------------
input <- copy(forecast_df)
input[, value:=(1-pi)]

# -------- map 
library(sf)
temp <- input[!pty_name %in% sel_pty]
temp <- temp[,.(value=mean(value)), by=c("lans_kommun_kod")]
temp[, pty_name:="other"]

sel <- grep("lans_kommun|value|pty_name", names(input), value = T)
map_df <- input[ pty_name %in% sel_pty, ..sel]
map_df <- rbind(map_df,temp)

kommun_shp <- st_read(file.path(getwd(), "input_data/alla_kommuner.shp"))
kommun_shp$KOM <- as.numeric(kommun_shp$KOM)

# all municipalities from shapefile
all_kommun <- unique(kommun_shp$KOM)
# all parties in map_df
all_party <- unique(map_df$pty_name)

# complete grid
full_grid <- CJ(
  lans_kommun_kod = all_kommun,
  pty_name = all_party,
  unique = TRUE
)

# join predictions
map_df <- merge(
  full_grid,
  map_df,
  by = c("lans_kommun_kod", "pty_name"),
  all.x = TRUE
)

# replace missing predictions with zero
# map_df[is.na(value), value := 0]

# attach geometry
map_df <- merge(
  kommun_shp,
  map_df,
  by.x = "KOM",
  by.y = "lans_kommun_kod",
  all.x = TRUE
)

# Relevel pty_short by lvl
map_df <- map_df %>%
  ungroup() %>%
  mutate(
    pty_name = factor(
      pty_name,
      levels = c(sel_pty, "other")
    )
  )

ggplot(map_df ) + #|> filter(pty_name!="other")
  geom_sf(aes(fill = value*100), color = NA) +
  facet_wrap(.~pty_name,  ncol=3,strip.position = "left") +
  scale_fill_viridis_c(
    option = "viridis",
    direction = -1,        # <-- reverse color order
        na.value = "grey80",
    name = "Prob. non-zero vote \n in %"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.major = element_blank(),   # remove major grid
    panel.grid.minor = element_blank(),   # remove minor grid
    panel.background = element_rect(fill = "#d5f3fe", color = NA),
    axis.text = element_blank(),          # optional: hide axis numbers
    axis.ticks = element_blank(),         # optional: hide axis ticks
    axis.title = element_blank(),          # optional: remove axis titles
    legend.position ="right"
  ) +
  coord_sf(expand = TRUE)
ggsave(file.path(outdir, "forecast_map_prob_vote.png"), width=8, height = 9)


#-----------------------------------------
# map forecast mu, expected vote counts
#-----------------------------------------
input <- copy(forecast_df)
input[, value:=(mu)]

# -------- map 
library(sf)
temp <- input[!pty_name %in% sel_pty]
temp <- temp[,.(value=sum(value)), by=c("lans_kommun_kod")]
temp[, pty_name:="other"]

sel <- grep("lans_kommun|value|pty_name", names(input), value = T)
map_df <- input[ pty_name %in% sel_pty, ..sel]
map_df <- rbind(map_df,temp)

# clip
clip_val <- quantile(map_df$value, probs = .99, na.rm = T) |> as.numeric()
map_df[, value:=fifelse(value>clip_val, clip_val, value)]

kommun_shp <- st_read(file.path(getwd(), "input_data/alla_kommuner.shp"))
kommun_shp$KOM <- as.numeric(kommun_shp$KOM)

# all municipalities from shapefile
all_kommun <- unique(kommun_shp$KOM)
# all parties in map_df
all_party <- unique(map_df$pty_name)

# complete grid
full_grid <- CJ(
  lans_kommun_kod = all_kommun,
  pty_name = all_party,
  unique = TRUE
)

# join predictions
map_df <- merge(
  full_grid,
  map_df,
  by = c("lans_kommun_kod", "pty_name"),
  all.x = TRUE
)

# replace missing predictions with zero
# map_df[is.na(value), value := 0]

# attach geometry
map_df <- merge(
  kommun_shp,
  map_df,
  by.x = "KOM",
  by.y = "lans_kommun_kod",
  all.x = TRUE
)

# Relevel pty_short by lvl
map_df <- map_df %>%
  ungroup() %>%
  mutate(
    pty_name = factor(
      pty_name,
      levels = c(sel_pty, "other")
    )
  )

ggplot(map_df ) + #|> filter(pty_name!="other")
  geom_sf(aes(fill = value), color = NA) +
  facet_wrap(.~pty_name, ncol=3, strip.position = "left") +
  scale_fill_viridis_c(
    option = "viridis",
    direction = -1,        # <-- reverse color order
        na.value = "grey80",
    name = "NB expected counts"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.major = element_blank(),   # remove major grid
    panel.grid.minor = element_blank(),   # remove minor grid
    panel.background = element_rect(fill = "#d5f3fe", color = NA),
    axis.text = element_blank(),          # optional: hide axis numbers
    axis.ticks = element_blank(),         # optional: hide axis ticks
    axis.title = element_blank(),          # optional: remove axis titles
    legend.position ="right"
  ) +
  coord_sf(expand = TRUE)
ggsave(file.path(outdir, "forecast_map_mu.png"), width=8, height = 9)


#-----------------------------------------
# map forecast uncertainty of expected vote counts
#-----------------------------------------
input <- copy(forecast_df)
input[, value:=log((mu + alpha*mu^2)/mu)]

# -------- map 
library(sf)
temp <- input[!pty_name %in% sel_pty]
temp <- temp[,.(value=mean(value)), by=c("lans_kommun_kod")]
temp[, pty_name:="other"]

sel <- grep("lans_kommun|value|pty_name", names(input), value = T)
map_df <- input[ pty_name %in% sel_pty, ..sel]
map_df <- rbind(map_df,temp)

# # clip
# clip_val <- quantile(map_df$value, probs = .99, na.rm = T) |> as.numeric()
# map_df[, value:=fifelse(value>clip_val, clip_val, value)]

kommun_shp <- st_read(file.path(getwd(), "input_data/alla_kommuner.shp"))
kommun_shp$KOM <- as.numeric(kommun_shp$KOM)

# all municipalities from shapefile
all_kommun <- unique(kommun_shp$KOM)
# all parties in map_df
all_party <- unique(map_df$pty_name)

# complete grid
full_grid <- CJ(
  lans_kommun_kod = all_kommun,
  pty_name = all_party,
  unique = TRUE
)

# join predictions
map_df <- merge(
  full_grid,
  map_df,
  by = c("lans_kommun_kod", "pty_name"),
  all.x = TRUE
)

# replace missing predictions with zero
# map_df[is.na(value), value := 0]

# attach geometry
map_df <- merge(
  kommun_shp,
  map_df,
  by.x = "KOM",
  by.y = "lans_kommun_kod",
  all.x = TRUE
)

# Relevel pty_short by lvl
map_df <- map_df %>%
  ungroup() %>%
  mutate(
    pty_name = factor(
      pty_name,
      levels = c(sel_pty, "other")
    )
  )

ggplot(map_df ) + #|> filter(pty_name!="other")
  geom_sf(aes(fill = value), color = NA) +
  facet_wrap(.~pty_name,  ncol=3, strip.position = "left") +
  scale_fill_viridis_c(
    option = "viridis",
    direction = -1,        # <-- reverse color order
    na.value = "grey80",
    name = "Log NB variance-to-mean ratio"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.major = element_blank(),   # remove major grid
    panel.grid.minor = element_blank(),   # remove minor grid
    panel.background = element_rect(fill = "#d5f3fe", color = NA),
    axis.text = element_blank(),          # optional: hide axis numbers
    axis.ticks = element_blank(),         # optional: hide axis ticks
    axis.title = element_blank(),          # optional: remove axis titles
    legend.position ="right"
  ) +
  coord_sf(expand = TRUE)
ggsave(file.path(outdir, "forecast_map_var_mu.png"), width=8, height = 8)

# #-----------------------------------------
# #  Map alpha dispersion, uncertainty
# #-----------------------------------------
# library(sf)

# input <- copy(forecast_df)
# input[, value:=alpha]
# input <- input[time==2026,  .(lans_kommun_kod, kommun_name, value)]

# # -------- map 
# library(sf)
# temp <- input
# temp <- unique(
#   temp,
#   by = c("lans_kommun_kod", "kommun_name")
# )
# map_df <- temp

# kommun_shp <- st_read(file.path(getwd(), "input_data/alla_kommuner.shp"))
# kommun_shp$KOM <- as.numeric(kommun_shp$KOM)

# # attach geometry
# map_df <- merge(
#   kommun_shp,
#   map_df,
#   by.x = "KOM",
#   by.y = "lans_kommun_kod",
#   all.x = TRUE
# )

# ggplot(map_df) +
#   geom_sf(aes(fill = value), color = "grey60") +
#   scale_fill_viridis_c(
#     option = "viridis",
#     direction = -1,        # <-- reverse color order
#     name = "Dispersion α"
#   ) +
#   theme_minimal(base_size = 14) +
#   theme(
#     panel.grid.major = element_blank(),   # remove major grid
#     panel.grid.minor = element_blank(),   # remove minor grid
#     panel.background = element_rect(fill = "#d5f3fe", color = NA),
#     axis.text = element_blank(),          # optional: hide axis numbers
#     axis.ticks = element_blank(),         # optional: hide axis ticks
#     axis.title = element_blank()          # optional: remove axis titles
#   ) 
# ggsave(file.path(outdir, "alpha_map.png"), width=8, height = 8)

# map_df |> ungroup() |> arrange(-value) |> select(NAMN_KOM, value) |> 
#   sf::st_drop_geometry() |> as.data.table()


#-----------------------------------------
#  plot alpha distribution
#-----------------------------------------
dt_alpha <- forecast_df[time==2026,  .(lans_kommun_kod, kommun_name, alpha)]
dt_alpha <- unique(
  dt_alpha,
  by = c("lans_kommun_kod", "kommun_name")
)

# ---- predictive distribution for each kommun, given its expected vote level and its volatility
dt_k <- forecast_df[
  time==2026,
  .(mu_hat    = mean(mu, na.rm = TRUE)),
  by = .(lans_kommun_kod, kommun_name)
]

dt_k <- merge(dt_k, dt_alpha, all = T)

# compute thresholds
alpha_qs <- c(0.001, seq(0.25, 0.75, by = 0.25), 0.999)
alpha_targets <- quantile(dt_k$alpha, alpha_qs, na.rm = TRUE)

alpha_targets_dt <- data.table(
  quantile = names(alpha_targets),
  q_value  = as.numeric(alpha_targets)
)

# For each quantile, pick the closest kommun
dt_quant_kommun <- alpha_targets_dt %>%
  rowwise() %>%
  do({
    qv <- .$q_value
    k  <- dt_k %>%
      mutate(dist = abs(alpha - qv)) %>%
      arrange(dist) %>%
      slice(1)
    cbind(k, quantile = .$quantile)
  }) %>%
  ungroup()

# Create readable labels
dt_quant_kommun <- dt_quant_kommun %>%
  mutate(
    label = paste0(
      "α=", round(alpha, 3),
      " | μ=", round(mu_hat, 0),
      " | ", kommun_name
    ),
    group = paste0("α quantile ", quantile)
  )

# function simulate NB
nb_pmf <- function(x, mu, alpha) {
  r <- 1 / alpha
  p <- r / (r + mu)
  dnbinom(x, size = r, prob = p)
}

dist_alpha <- dt_quant_kommun %>%
  rowwise() %>%
  do({
    mu  <- .$mu_hat
    a   <- .$alpha
    x   <- seq(0, round(mu * 3))
    y   <- nb_pmf(x, mu = mu, alpha = a)
    data.frame(
      kommun_name = .$kommun_name,
      quantile    = .$quantile,
      label       = .$label,
      group       = .$group,
      x = x,
      x_rel = x / mu,
      prob = y,
      mu_hat = mu,
      alpha = a
    )
  }) %>%
  ungroup()

label_df <- dist_alpha %>%
  group_by(kommun_name) %>%
  slice_max(prob, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(
    x = x_rel * 0.8,
    y = prob * 0.95
  )

ggplot(dist_alpha,
       aes(x = x_rel, y = prob, group = kommun_name)) +
  geom_line(aes(color=group), alpha=0.68) +
  ggrepel::geom_text_repel(
    data = label_df,
    aes(x = x, y = y, label = label, color=group), show.legend = F,
    size = 4, direction = "x"
  ) +
  theme_minimal(base_size = 14) +
  labs(
    x = "Normalized: Votes / μ",
    y = "Probability mass",
    color = "α quantile"
  ) +
  scale_color_viridis_d()
ggsave(file.path(outdir, "alpha_dist_kommun.png"), width = 8, height = 6)
