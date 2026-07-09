
# ------------------------------------
# config.
# ------------------------------------

# If running in RStudio, set working dir to script location (optional)
if (requireNamespace("rstudioapi", quietly = TRUE) &&
  rstudioapi::isAvailable()) {
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  }

pacman::p_load(tidyverse,data.table, keras, tfruns)

# set virtual env.
use_virtualenv("~/v_environment/r-tf", required = TRUE)

# ---- Determinism toggles (set before loading tensorflow/keras) ----
Sys.setenv(PYTHONHASHSEED = "0")      # Python hash seed
Sys.setenv(TF_DETERMINISTIC_OPS = "1")# request deterministic TF ops (TF>=2.13)
set.seed(2026)                        # R's RNG

tensorflow::tf$random$set_seed(2026)

# ---- input enet and zinb functions ----
source("functions_zinb_gate_2.r")
source("preproc_zinb_gate_2.r") # this will load transformed data and splits for training

# -------------------------------------------------------
# trvate explanatory 
# -------------------------------------------------------
loaded <- load_zinb_model(file.path(getwd(), "trvate_mod"))
model <- loaded$model
scalers <- loaded$scalers
meta <- loaded$meta
index_maps <- loaded$index_maps
with_inter_final <- loaded$with_inter_final
training_hparams <- loaded$training_hparams

# ------------------------ list layer names
sapply(model$layers, function(x) x$name)

# ------------------------ gating values
get_gate_values <- function(model, gate_name) {
  layer <- model$layers[[ which(vapply(model$layers, function(x) x$name == gate_name, logical(1))) ]]
  gv <- as.numeric(layer$weights[[1]]$numpy())  # gate_vector before sigmoid
  sigmoid <- function(x) 1 / (1 + exp(-x))
  g <- sigmoid(gv)
  data.table::data.table(feature = meta$num_features, gate_raw = gv, gate = g)
}

gates_zero <- get_gate_values(model, "gates_zero")
print(gates_zero[order(-gate)])

gates_cnt <- get_gate_values(model, "gates_count")
print(gates_cnt[order(-gate)])

# ------------------------ ablation
# ---- helper x num feature
ablate_numeric_feature <- function(model, x_data, feature_name,
                                   num_features, use_softplus_mu, y_true) {

  x0 <- x_data
  k <- match(feature_name, num_features)
  if (is.na(k)) stop("Feature not found: ", feature_name)

  # Zero out (standard practice in ablation)
  x0[[1]][, k] <- 0

  out <- predict_zinb(model, x = x0, y_true = y_true,
                      use_softplus_mu = use_softplus_mu)

  out$pred
}

# ---- helper z bin feature
ablate_bin_feature <- function(model, x_data, feature_name,
                               Z_bin_features, use_softplus_mu, y_true) {

  x0 <- x_data
  k <- match(feature_name, Z_bin_features)
  if (is.na(k)) stop("Binary feature not found: ", feature_name)

  # Zero out binary column
  x0[[3]][, k] <- 0

  out <- predict_zinb(model, x = x0, y_true = y_true,
                      use_softplus_mu = use_softplus_mu)
  out$pred
}

# ---- helper embedding
ablate_embedding <- function(model, x_data, type = c("party","kommun","interact"),
                             use_softplus_mu, y_true) {

  type <- match.arg(type)
  x0 <- x_data

  if (type == "party") {
    x0[[5]][] <- 1   # collapse all party IDs
  } else if (type == "kommun") {
    x0[[6]][] <- 1
  } else if (type == "interact") {
    # Only ablate if interaction embedding exists (7th tensor)
    if (length(x0) == 7L) x0[[7]][] <- 1
  }

  out <- predict_zinb(model, x = x0, y_true = y_true,
                      use_softplus_mu = use_softplus_mu)
  out$pred
}

# ---- master ablation function
run_full_ablation <- function(model,
                              x_data,
                              y_true,
                              num_features, Z_bin_features,
                              with_inter,
                              use_softplus_mu = FALSE) {

  # baseline performance
  base <- predict_zinb(model,
                       x_data = x_data,
                       y_true = y_true,
                       with_inter = with_inter,
                       use_softplus_mu = use_softplus_mu)

  base_r2  <- base$r2_corr[1]
  base_mae <- mean(abs(base$pred - y_true))

  results <- list()

  # ---- numeric features ----
  for (f in num_features) {
    preds <- ablate_numeric_feature(model, x_data, f,
                                    num_features, use_softplus_mu, y_true)
    r2  <- tryCatch(cor(preds, y_true)^2, error=function(e) NA_real_)
    mae <- mean(abs(preds - y_true))
    results[[f]] <- tibble::tibble(
      type = "numeric",
      feature = f,
      r2 = r2,
      mae = mae,
      dr2 = r2/base_r2 - 1,
      dmae = mae/base_mae - 1
    )
  }

  # ---- binary ZI features ----
  for (f in Z_bin_features) {
    preds <- ablate_bin_feature(model, x_data, f,
                                Z_bin_features, use_softplus_mu, y_true)
    r2  <- tryCatch(cor(preds, y_true)^2, error=function(e) NA_real_)
    mae <- mean(abs(preds - y_true))
    results[[paste0("bin_", f)]] <- tibble::tibble(
      type = "binary",
      feature = f,
      r2 = r2,
      mae = mae,
      dr2 = r2/base_r2 - 1,
      dmae = mae/base_mae - 1
    )
  }

  # ---- embeddings ----
  emb_types <- c("party", "kommun")
  if (with_inter) emb_types <- c(emb_types, "interact")

  for (emb in emb_types) {
    preds <- ablate_embedding(model, x_data, emb,
                               use_softplus_mu, y_true)
    r2  <- tryCatch(cor(preds, y_true)^2, error=function(e) NA_real_)
    mae <- mean(abs(preds - y_true))
    results[[paste0("emb_", emb)]] <- tibble::tibble(
      type = "embedding",
      feature = emb,
      r2 = r2,
      mae = mae,
      dr2 = r2/base_r2 - 1,
      dmae = mae/base_mae - 1
    )
  }

  # Combine & rank
  dplyr::bind_rows(results) 
}

# ---- run
temp <- run_full_ablation(
  loaded$model,
  x_data = loaded$training_data$x,
  y_true = loaded$training_data$y,
  num_features = loaded$meta$num_features,
  Z_bin_features = loaded$meta$Z_bin_features,
  with_inter = with_inter_final,
  use_softplus_mu = loaded$training_hparams$use_softplus_mu
) |> as.data.table()
abla <- temp[, c("dr2", "dmae"):=lapply(.SD, round, 3), .SDcols = c("dr2", "dmae")]
print(abla[order(-dmae)])


# ------------------------ Elasticity
# ---- Numeric feature elasticity
elasticity_numeric <- function(model, x_data, feature_name,
                               num_features, y_base, use_softplus_mu,
                               delta = 0.1, eps = 1e-9) {

  k <- match(feature_name, num_features)
  x1 <- x_data
  x1[[1]][, k] <- x1[[1]][, k] * (1 + delta)

  pred_full <- predict(model, x1)
  params <- extract_params(pred_full, use_softplus_mu = use_softplus_mu)
  mu     <- params$mu
  alpha  <- params$alpha
  pi     <- params$pi
  pred1   <- (1 - pi) * mu                       # expected counts

  # elasticity = ((p1 - p0)/p0) / delta
  elas <- ((pred1 - y_base) / (y_base + eps)) / delta

  tibble::tibble(
    feature = feature_name,
    type = "numeric",
    elasticity = elas,
    mean_elasticity = mean(elas, na.rm = TRUE),
    p10 = quantile(elas, 0.10, na.rm=TRUE),
    p50 = quantile(elas, 0.50, na.rm=TRUE),
    p90 = quantile(elas, 0.90, na.rm=TRUE)
  ) |> select(-elasticity) |> distinct()
}


# ---- Z‑binary feature elasticity
# Z-bin features often represent “presence indicators”. We perturb them by turning them off (set to 0).
elasticity_bin <- function(model, x_data, feature_name,
                           Z_bin_features, y_base, use_softplus_mu, eps=1e-9) {

  k <- match(feature_name, Z_bin_features)
  x1 <- x_data
  x1[[3]][, k] <- 0

  pred_full <- predict(model, x1)
  params <- extract_params(pred_full, use_softplus_mu = use_softplus_mu)
  mu     <- params$mu
  alpha  <- params$alpha
  pi     <- params$pi
  pred1   <- (1 - pi) * mu                       # expected counts

  # treat binary "shock" as Δx = 1 → elasticity equals %Δy
  elas <- (pred1 - y_base) / (y_base + eps)

  tibble::tibble(
    feature = feature_name,
    type = "binary",
    elasticity = elas,
    mean_elasticity = mean(elas, na.rm=TRUE),
    p10 = quantile(elas, 0.10, na.rm=TRUE),
    p50 = quantile(elas, 0.50, na.rm=TRUE),
    p90 = quantile(elas, 0.90, na.rm=TRUE)
  ) |> select(-elasticity) |> distinct()
}

# ---- Master Elasticity Function 
run_full_elasticity <- function(model,
                                x_data,
                                num_features, Z_bin_features,
                                with_inter,
                                use_softplus_mu = FALSE) {

  # baseline predictions
  pred_full <- predict(model, x_data)
  params <- extract_params(pred_full, use_softplus_mu = use_softplus_mu)
  mu     <- params$mu
  alpha  <- params$alpha
  pi     <- params$pi
  base_pred   <- (1 - pi) * mu                       # expected counts

  out <- list()

  # ---- numeric features ----
  for (f in num_features) {
    out[[paste0("num_", f)]] <- elasticity_numeric(
      model, x_data, f, num_features,
      y_base = base_pred,
      use_softplus_mu = use_softplus_mu
    )
  }

  # ---- binary features ----
  for (f in Z_bin_features) {
    out[[paste0("bin_", f)]] <- elasticity_bin(
      model, x_data, f, Z_bin_features,
      y_base = base_pred,
      use_softplus_mu = use_softplus_mu
    )
  }

  # Return stacked tibble
  dplyr::bind_rows(out) 
}

# ---- run
temp <- run_full_elasticity(
  loaded$model,
  x_data = loaded$training_data$x,
  num_features = loaded$meta$num_features,
  Z_bin_features = loaded$meta$Z_bin_features,
  with_inter = with_inter_final,
  use_softplus_mu = loaded$training_hparams$use_softplus_mu
) |> as.data.table()

elas <- copy(temp)
num_cols <- names(elas)[sapply(elas, is.numeric)]
elas[, (num_cols) := lapply(.SD, round, 3), .SDcols = num_cols]
print(elas[order(-abs(mean_elasticity))])


# ------------------------ combine gates, abla, elas
temp <- merge(abla, elas, by=c("feature", "type"), all=T)
selcol <- grep("feature|type|dmae|elastici", names(temp), value=T)
temp <- temp[, ..selcol]

temp_gates <- merge(gates_zero, gates_cnt, by=c("feature"), all=T, suffixes = c("_zero", "_cnt"))
selcol <- grep("_raw", names(temp_gates), value=T)
temp_gates <- temp_gates[, -..selcol]

combine <- merge(temp, temp_gates, by="feature", all=T)
combine[order(-dmae)]


# ------------------------ Analyzie embedding
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
emb_party_mat <- get_embedding_matrix(model, index = 1)  
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
  mutate(main=ifelse(pty_short %in% c("l","s","sd","m","c","v","mp","fi"), 1, 0))

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
ggsave(file.path(getwd(), "results/emb_party_pca.png"), width=8, height = 8)


# ------------------------ Analyze alpha dispersion
loaded <- load_zinb_model(file.path(getwd(), "trvate_mod"))

dt_alpha <- predict_zinb(
  model,
  x_data = loaded$training_data$x, 
  y_true = loaded$training_data$y, use_softplus_mu = loaded$training_hparams$use_softplus_mu,
  with_inter = loaded$with_inter_final                  # you can omit this; it auto-infers from model$inputs
)

temp <- rbind(
  dt[loaded$splits$tr_rows],
  dt[loaded$splits$va_rows],
  dt[loaded$splits$te_rows]
)

temp[, alpha := dt_alpha$alpha]
temp[, mu := dt_alpha$mu]
temp[, pi := dt_alpha$pi]
temp[, pred := dt_alpha$pred]

dt_alpha <- copy(temp)

# ---- predictive distribution for each kommun, given its expected vote level and its volatility
dt_k <- dt_alpha[
  , .(
    alpha_hat = mean(alpha, na.rm = TRUE),
    mu_hat    = mean(mu, na.rm = TRUE)
  ),
  by = kommun_name
]

# compute thresholds
alpha_qs <- c(0.01, seq(0.25, 0.75, by = 0.25), 0.99)
alpha_targets <- quantile(dt_k$alpha_hat, alpha_qs, na.rm = TRUE)

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
      mutate(dist = abs(alpha_hat - qv)) %>%
      arrange(dist) %>%
      slice(1)
    cbind(k, quantile = .$quantile)
  }) %>%
  ungroup()

# Create readable labels
dt_quant_kommun <- dt_quant_kommun %>%
  mutate(
    label = paste0(
      "α=", round(alpha_hat, 3),
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
    a   <- .$alpha_hat
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
      alpha_hat = a
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
ggsave(file.path(getwd(), "results/alpha_dist_kommun.png"), width = 8, height = 6)

# ---- map alpha
library(sf)

alpha_kommun <- dt_alpha[, .(
  mean_alpha = mean(alpha),
  median_alpha = median(alpha),
  sd_alpha = sd(alpha)
), by = kommun_factor]

alpha_kommun <- merge(
  alpha_kommun,
  unique(dt[, .(kommun_factor, kommun_name, lans_kommun_kod)]),
  by = "kommun_factor",
  all.x = TRUE
)
alpha_kommun[order(-mean_alpha)][1:20]

kommun_shp <- st_read(file.path(getwd(), "input_data/alla_kommuner.shp"))
kommun_shp$KOM <- as.numeric(kommun_shp$KOM)
map_df <- merge(kommun_shp, alpha_kommun,
                by.x="KOM", by.y="lans_kommun_kod")

ggplot(map_df) +
  geom_sf(aes(fill = mean_alpha), color = "grey60") +
  scale_fill_viridis_c(
    option = "viridis",
    direction = -1,        # <-- reverse color order
    name = "Dispersion α"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.major = element_blank(),   # remove major grid
    panel.grid.minor = element_blank(),   # remove minor grid
    panel.background = element_rect(fill = "#d5f3fe", color = NA),
    axis.text = element_blank(),          # optional: hide axis numbers
    axis.ticks = element_blank(),         # optional: hide axis ticks
    axis.title = element_blank()          # optional: remove axis titles
  ) 
ggsave(file.path(getwd(), "results/alpha_map.png"), width=8, height = 8)

map_df |> ungroup() |> arrange(-mean_alpha) |> select(NAMN_KOM, mean_alpha) |> 
  sf::st_drop_geometry() |> as.data.table()
