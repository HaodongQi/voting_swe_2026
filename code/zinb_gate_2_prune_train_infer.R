
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


#######################################################################################
# -------- pre process
#######################################################################################

f <- list.files(file.path(getwd(), "input_data"), full.names = TRUE)
f <- grep("all_year_clean", f, value = TRUE)
dt <- fread(f)

# offset
stopifnot("tot_elig_voter" %in% names(dt))
dt[, offset_log := log(pmax(1, tot_elig_voter))]

# ---- lag flags & stability features
stopifnot("lag_votes" %in% names(dt))                          # ensure lag_votes exists
dt[, has_lag_vote := as.integer(!is.na(lag_votes))]            # <-- exists, not > 0
if (!"lag_log_votes" %in% names(dt)) {
  dt[, lag_log_votes := ifelse(is.na(lag_votes), NA_real_, log1p(pmax(0, lag_votes)))]
}
# If you already computed lag_votes_pct (lag / current-year eligibles), we'll use it below if present.

# --- define inputs (numeric & ZI)
num_features <- unique(c(
  grep("^cand_|^sim_|^vot_", names(dt), value = TRUE),
  "lag_votes",
  intersect(c("lag_votes_pct","lag_log_votes"), names(dt))     # include if present
))
Z_features <- num_features

# binary flags for ZI
Z_bin_features <- grep("^has_", names(dt), value = TRUE)

# ---- Targets and offset
y <- dt$votes
offset <- matrix(dt$offset_log, ncol = 1)

# -------------------------
# Party / kommun embeddings
# -------------------------
# Prefer pty_code for stable indexing; map to dense 1..P
if (!"pty_code" %in% names(dt)) stop("pty_code missing in input files.")
dt[, pty_code := as.integer(pty_code)]
# UNK for NA pty_code (optional, not strictly needed if none are NA)
UNK_CODE <- ifelse(all(is.na(dt$pty_code)), 99999, max(dt$pty_code, na.rm = TRUE) + 1L)
dt[is.na(pty_code), pty_code := UNK_CODE]

dt[, party_factor := as.integer(factor(pty_code, levels = sort(unique(pty_code))))]  # dense 1..P
n_party <- dt[, uniqueN(party_factor)]
party_idx <- matrix(dt$party_factor, ncol = 1)

# Kommun embedding id
dt[, kommun_factor := as.integer(factor(lans_kommun_kod))]
n_kommun <- dt[, uniqueN(kommun_factor)]
kommun_idx <- matrix(dt$kommun_factor, ncol = 1)

# Party–kommun interaction embedding (OPTIONAL; keep if your builder expects it)
dt[, party_kommun := paste0(party_factor, "|", kommun_factor)]
dt[, party_kommun_factor := as.integer(factor(party_kommun))]
n_interact <- dt[, uniqueN(party_kommun_factor)]
interact_idx <- matrix(dt$party_kommun_factor, ncol = 1)

# -------------------------
# save preprocssed data
# -------------------------
fwrite(dt, "preprocessed_data.csv")

# ------------------------------------
# 60/20/20 kommun split (panel-aware)
# ------------------------------------
set.seed(2026L)
all_kommun <- sort(unique(dt$kommun_factor))
n_total <- length(all_kommun)
n_tr    <- max(1L, round(n_total * 0.60))
n_va    <- max(1L, round(n_total * 0.20))

tr_kommun_ids <- sort(sample(all_kommun, n_tr))
rem_kommun    <- setdiff(all_kommun, tr_kommun_ids)
va_kommun_ids <- sort(sample(rem_kommun, n_va))
te_kommun_ids <- sort(setdiff(rem_kommun, va_kommun_ids))

tr_rows <- dt$kommun_factor %in% tr_kommun_ids
va_rows <- dt$kommun_factor %in% va_kommun_ids
te_rows <- dt$kommun_factor %in% te_kommun_ids

# --- Add year masks (assumes your year column is "time"; change to "val_yr" if needed)
stopifnot("time" %in% names(dt) )
year_col <- if ("time" %in% names(dt)) "time" 

is_hist <- dt[[year_col]] %in% c(2014L, 2018L, 2022L)   # training/eval cycles
is_2026 <- dt[[year_col]] == 2026L                      # forecast cycle

# Intersect with kommun split so 2026 doesn't leak into tr/va/te:
tr_rows <- tr_rows & is_hist
va_rows <- va_rows & is_hist
te_rows <- te_rows & is_hist

fo_rows <- is_2026  # forecast rows (all kommun & parties present in 2026)

# ------------------------------------
# Fit scalers on TRAIN ONLY (NA-safe), transform all splits
# ------------------------------------
scale_fit  <- function(M) {
  mu <- colMeans(M, na.rm = TRUE)
  sd <- apply(M, 2, sd, na.rm = TRUE)
  list(mu = mu, sd = pmax(sd, 1e-9))
}

scale_apply <- function(M, fit) {
  if (is.null(dim(M))) M <- matrix(M, ncol = 1L)
  M0 <- M
  # impute NA with TRAIN means
  for (j in seq_len(ncol(M0))) {
    idx_na <- is.na(M0[, j])
    if (any(idx_na)) M0[idx_na, j] <- fit$mu[j]
  }
  sweep(sweep(M0, 2, fit$mu, "-"), 2, fit$sd, "/")
}

X_num_all <- as.matrix(dt[, ..num_features])
Z_num_all <- as.matrix(dt[, ..Z_features])
if (length(Z_bin_features)) {
  Z_bin_all <- as.matrix(dt[, ..Z_bin_features])
} else {
  Z_bin_all <- matrix(0, nrow(dt), 0)
}

fit_num <- scale_fit(X_num_all[tr_rows, , drop = FALSE])
fit_Z   <- scale_fit(Z_num_all[tr_rows, , drop = FALSE])

X_num_tr <- scale_apply(X_num_all[tr_rows, , drop = FALSE], fit_num)
X_num_va <- scale_apply(X_num_all[va_rows, , drop = FALSE], fit_num)
X_num_te <- scale_apply(X_num_all[te_rows, , drop = FALSE], fit_num)
X_num_fo <- scale_apply(X_num_all[fo_rows, , drop = FALSE], fit_num)

Z_num_tr <- scale_apply(Z_num_all[tr_rows, , drop = FALSE], fit_Z)
Z_num_va <- scale_apply(Z_num_all[va_rows, , drop = FALSE], fit_Z)
Z_num_te <- scale_apply(Z_num_all[te_rows, , drop = FALSE], fit_Z)
Z_num_fo <- scale_apply(Z_num_all[fo_rows, , drop = FALSE], fit_Z)

Z_bin_tr <- Z_bin_all[tr_rows, , drop = FALSE]
Z_bin_va <- Z_bin_all[va_rows, , drop = FALSE]
Z_bin_te <- Z_bin_all[te_rows, , drop = FALSE]
Z_bin_fo <- Z_bin_all[fo_rows, , drop = FALSE]
storage.mode(Z_bin_tr) <- "double"
storage.mode(Z_bin_va) <- "double"
storage.mode(Z_bin_te) <- "double"
storage.mode(Z_bin_fo) <- "double"

# Targets and other inputs
y_tr <- y[tr_rows]; y_va <- y[va_rows]; y_te <- y[te_rows]; y_fo <- y[fo_rows]
off_tr <- offset[tr_rows, , drop = FALSE]
off_va <- offset[va_rows, , drop = FALSE]
off_te <- offset[te_rows, , drop = FALSE]
off_fo <- offset[fo_rows, , drop = FALSE]

party_tr <- party_idx[tr_rows, , drop = FALSE]
party_va <- party_idx[va_rows, , drop = FALSE]
party_te <- party_idx[te_rows, , drop = FALSE]
party_fo <- party_idx[fo_rows, , drop = FALSE]

kommun_tr <- kommun_idx[tr_rows, , drop = FALSE]
kommun_va <- kommun_idx[va_rows, , drop = FALSE]
kommun_te <- kommun_idx[te_rows, , drop = FALSE]
kommun_fo <- kommun_idx[fo_rows, , drop = FALSE]

interact_tr <- interact_idx[tr_rows, , drop = FALSE]
interact_va <- interact_idx[va_rows, , drop = FALSE]
interact_te <- interact_idx[te_rows, , drop = FALSE]
interact_fo <- interact_idx[fo_rows, , drop = FALSE]

# ---- Pack Keras lists: train, val, test
make_x <- function(with_inter, X_num, Z_num, Z_bin, off, party, kommun, interact) {
  if (with_inter) list(X_num, Z_num, Z_bin, off, party, kommun, interact)
  else            list(X_num, Z_num, Z_bin, off, party, kommun)
}

train_data <- list(
  x_with = make_x(
    TRUE,
    X_num_tr, Z_num_tr, Z_bin_tr, off_tr,
    party_tr, kommun_tr, interact_tr
  ),
  x_wo = make_x(
    FALSE,
    X_num_tr, Z_num_tr, Z_bin_tr, off_tr,
    party_tr, kommun_tr, interact_tr
  ),
  y = y_tr
)
val_data <- list(
  x_with = make_x(
    TRUE,
    X_num_va, Z_num_va, Z_bin_va, off_va,
    party_va, kommun_va, interact_va
  ),
  x_wo = make_x(
    FALSE,
    X_num_va, Z_num_va, Z_bin_va, off_va,
    party_va, kommun_va, interact_va
  ),
  y = y_va
)
test_data <- list(
  x_with = make_x(
    TRUE,
    X_num_te, Z_num_te, Z_bin_te, off_te,
    party_te, kommun_te, interact_te
  ),
  x_wo = make_x(
    FALSE,
    X_num_te, Z_num_te, Z_bin_te, off_te,
    party_te, kommun_te, interact_te
  ),
  y = y_te
)
fo_data <- list(
  x_with = make_x(
    TRUE,
    X_num_fo, Z_num_fo, Z_bin_fo, off_fo,
    party_fo, kommun_fo, interact_fo
  ),
  x_wo = make_x(
    FALSE,
    X_num_fo, Z_num_fo, Z_bin_fo, off_fo,
    party_fo, kommun_fo, interact_fo
  ),
  y = y_fo
)


#######################################################################################
# ------------ Training functions
#######################################################################################

source(file.path(getwd(), "functions_zinb_gate_2_prune.R"))

#######################################################################################
# -------- Tuning
#######################################################################################

# ------------------------------------
# training control
# ------------------------------------

# Bias initialization for bounded vote-share μ
# μ = eligible_voters * sigmoid(eta)
# Therefore init bias should be logit(mean vote share)
exp_off_tr <- as.numeric(exp(off_tr))

rate_tr <- if (length(y_tr)) {
  mean(y_tr / pmax(1e-9, exp_off_tr), na.rm = TRUE)
} else {
  1e-6
}
# Bound away from 0 and 1 before logit
rate_tr <- pmin(pmax(rate_tr, 1e-6), 1 - 1e-6)
# b_mu0 is now logit vote share, not log count rate
b_mu0 <- qlogis(rate_tr)

p0_tr <- if (length(y_tr)) mean(y_tr == 0, na.rm = TRUE) else 0.5
b_pi0 <- qlogis(pmin(pmax(p0_tr, 1e-6), 1 - 1e-6))

# other control
use_gelu <- TRUE
use_softplus_mu <- FALSE
lr <- 1e-4

alpha_mode <- "group"          # kommun-specific α
alpha_clip <- c(-5, 5)
alpha_group_use <- c("kommun") # <-- kommun only
alpha_group_l2 <- 1e-6
alpha_global_init_log <- log(0.1)

# initial embedding dims (placeholders; tuner will pick best)
emb_dim_party <- 3L
emb_dim_kommun <- 8L
emb_dim_interact <- 1L   # set to 0 only if your builder skips interaction when 0


# # ------------------------------------
# # Coarse-to-fine tuning (two gates, no coupling)
# # ------------------------------------
# k_clear_session()
# set_global_seed(2026)

# cf <- coarse_to_fine_two_gates(
#   cnt_range = c(1e-6, 1e-1),
#   zi_range  = c(1e-6, 1e-1),
#   n_stage1  = 200,

#   top_k = 5,
#   factors = c(0.25, 1, 2),
#   n_max_stage2 = 200,

#   alpha_en_cnt = 0.5, alpha_en_zi = 0.5,
#   train_data = train_data, val_data = val_data,
#   num_features = num_features, Z_features = Z_features, Z_bin_features = Z_bin_features,
#   n_party = n_party, n_kommun = n_kommun, n_interact = n_interact,

#   w_zero_grid     = c(1),
#   emb_party_grid  = c(1, 2, 4, 8, 16),
#   emb_kommun_grid = c(1, 2, 4, 8, 16),
#   emb_inter_grid  = c(1),     # or c(0,1) only if builder supports zero-dim interaction

#   emb_dim_party = emb_dim_party,
#   emb_dim_kommun = emb_dim_kommun,
#   emb_dim_interact = emb_dim_interact,

#   lr = lr, seed = 2026,
#   use_gelu = use_gelu,
#   use_softplus_mu = use_softplus_mu,
#   init_log_mu_bias = b_mu0,
#   init_logit_pi_bias = b_pi0,

#   alpha_mode = alpha_mode,
#   alpha_clip = alpha_clip,
#   alpha_group_use = alpha_group_use,
#   alpha_group_l2  = alpha_group_l2,
#   alpha_global_init_log = alpha_global_init_log,

#   epochs_stage1 = 40, patience_stage1 = 3,
#   epochs_stage2 = 80, patience_stage2 = 6,
#   batch_size = 512, verbose_fit = 0,

#   allow_zero_interact = TRUE  
# )

# message(sprintf("Best: λ_cnt=%.4g λ_zi=%.4g | w_zero=%.2f | emb=%d/%d/%d | val_loss=%.6f",
#                 cf$best$lambda_cnt, cf$best$lambda_zi, cf$best$w_zero,
#                 cf$best$emb_party, cf$best$emb_kommun, cf$best$emb_inter,
#                 cf$best$val_loss))


#######################################################################################
# -------- trva training -> trva_prune training
#######################################################################################

# ------------------------------------
#  best params
# ------------------------------------
f <- list.files(file.path(getwd(), "tuner_results"), recursive = T, full.names = T)
f <- grep("best_", f, ignore.case = T, value = T)
best <- readRDS(f)

# ------------------------------------
# Combine TRAIN + VAL  
# ------------------------------------
X_num_trva <- rbind(X_num_tr, X_num_va)
Z_num_trva <- rbind(Z_num_tr, Z_num_va)
Z_bin_trva <- rbind(Z_bin_tr, Z_bin_va)

off_trva     <- rbind(off_tr, off_va)
party_trva   <- rbind(party_tr, party_va)
kommun_trva  <- rbind(kommun_tr, kommun_va)
interact_trva<- rbind(interact_tr, interact_va)

y_trva <- c(y_tr, y_va)

trva_data <- list(
  x_with = make_x(
    TRUE,
    X_num_trva, Z_num_trva, Z_bin_trva, off_trva,
    party_trva, kommun_trva, interact_trva
  ),
  x_wo = make_x(
    FALSE,
    X_num_trva, Z_num_trva, Z_bin_trva, off_trva,
    party_trva, kommun_trva, interact_trva
  ),
  y = y_trva
)


# ------------------------------------
# Refit soft model on TRAIN + VAL
# ------------------------------------
with_inter_final <- best$emb_inter > 0L
trva_x <- if (with_inter_final) trva_data$x_with else trva_data$x_wo

k_clear_session()
set_global_seed(2026)

soft_model <- build_zinb_nn(
  lambda_en_cnt = best$lambda_cnt,
  alpha_en_cnt  = 0.5,
  lambda_en_zi  = best$lambda_zi,
  alpha_en_zi   = 0.5,

  num_features   = num_features,
  Z_features     = Z_features,
  Z_bin_features = Z_bin_features,

  n_party    = n_party,
  n_kommun   = n_kommun,
  n_interact = n_interact,

  emb_dim_party    = best$emb_party,
  emb_dim_kommun   = best$emb_kommun,
  emb_dim_interact = best$emb_inter,

  lr = lr,
  seed = 2026,
  w_zero = best$w_zero,

  use_gelu = use_gelu,
  use_softplus_mu = use_softplus_mu,

  init_log_mu_bias = b_mu0,
  init_logit_pi_bias = b_pi0,

  alpha_mode = alpha_mode,
  alpha_clip = alpha_clip,
  alpha_group_use = alpha_group_use,
  alpha_group_l2  = alpha_group_l2,
  alpha_global_init_log = alpha_global_init_log
)

history_soft <- soft_model %>% fit(
  x = trva_x,
  y = trva_data$y,
  epochs = 200,
  batch_size = 512,
  verbose = 2,
  shuffle = FALSE,
  workers = 1,
  use_multiprocessing = FALSE,
  callbacks = list(
    callback_early_stopping(
      monitor = "loss",
      patience = 15,
      restore_best_weights = TRUE
    ),
    callback_reduce_lr_on_plateau(
      monitor = "loss",
      factor = 0.5,
      patience = 4,
      min_lr = 1e-5
    )
  )
)


# ------------------------------------
# Gate extraction
# ------------------------------------

get_gate_values <- function(model, gate_name, feature_names) {

  layer_idx <- which(vapply(
    model$layers,
    function(x) x$name == gate_name,
    logical(1)
  ))

  if (length(layer_idx) != 1L) {
    stop("Gate layer not found or not unique: ", gate_name)
  }

  layer <- model$layers[[layer_idx]]

  raw <- as.numeric(layer$weights[[1]]$numpy())
  gate <- 1 / (1 + exp(-raw))

  tibble::tibble(
    feature = feature_names,
    gate_raw = raw,
    gate = gate,
  ) |>
    dplyr::arrange(dplyr::desc(gate))
}

gates_count <- get_gate_values(
  soft_model,
  gate_name = "gates_count",
  feature_names = num_features
)

gates_zero <- get_gate_values(
  soft_model,
  gate_name = "gates_zero",
  feature_names = Z_features
)

gate_dir <- file.path(getwd(), "gate_results")
if (dir.exists(gate_dir)) unlink(gate_dir, recursive = TRUE)
dir.create(gate_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(gates_count, file.path(gate_dir, "gates_count.csv"), row.names = FALSE)
write.csv(gates_zero,  file.path(gate_dir, "gates_zero.csv"),  row.names = FALSE)

saveRDS(
  list(
    gates_count = gates_count,
    gates_zero = gates_zero
  ),
  file.path(gate_dir, "gate_values.rds")
)


# ------------------------------------
# Hard feature pruning
# ------------------------------------
select_features_from_gates <- function(gate_tbl,
                                       threshold = 0.55,
                                       min_keep = 5,
                                       max_keep = Inf,
                                       feature_col = "feature",
                                       gate_col = "gate") {

  # order by gate value
  gate_ranked <- gate_tbl |>
    dplyr::arrange(dplyr::desc(.data[[gate_col]])) |>
    dplyr::mutate(
      rank = dplyr::row_number(),
      passed_threshold = .data[[gate_col]] > threshold
    )

  # first select by threshold
  selected <- gate_ranked |>
    dplyr::filter(passed_threshold)

  # if too few pass threshold, keep top min_keep
  if (nrow(selected) < min_keep) {
    selected <- gate_ranked |>
      dplyr::slice_head(n = min_keep) |>
      dplyr::mutate(selection_rule = paste0("top_", min_keep, "_fallback"))
  } else {
    selected <- selected |>
      dplyr::mutate(selection_rule = paste0("gate_ge_", threshold))
  }

  # if too many pass threshold, cap at max_keep
  if (is.finite(max_keep) && nrow(selected) > max_keep) {
    selected <- selected |>
      dplyr::arrange(dplyr::desc(.data[[gate_col]])) |>
      dplyr::slice_head(n = max_keep) |>
      dplyr::mutate(selection_rule = paste0(selection_rule, "_capped_top_", max_keep))
  }

  list(
    selected_features = selected[[feature_col]],
    selected_table = selected,
    ranked_table = gate_ranked
  )
}

# set threshold
gate_threshold <- 0
# adjust final lambda depending on threshold
if(gate_threshold>0) prune_lambda_cnt <- 0 else prune_lambda_cnt <- best$lambda_cnt
if(gate_threshold>0) prune_lambda_zi <- 0 else prune_lambda_zi <- best$lambda_zi

num_features_pruned <- select_features_from_gates(
  gates_count,
  threshold = gate_threshold,
  min_keep = 1,
  max_keep = Inf
)

Z_features_pruned <- select_features_from_gates(
  gates_zero,
  threshold = gate_threshold,
  min_keep = 1,
  max_keep = Inf

)

feature_selection <- list(
  threshold = gate_threshold,

  num_features_original = num_features,
  Z_features_original   = Z_features,

  num_features_pruned = num_features_pruned,
  Z_features_pruned   = Z_features_pruned,

  gates_count = gates_count,
  gates_zero  = gates_zero,

  count_ranked_table = num_features_pruned$ranked_table,
  zero_ranked_table  = Z_features_pruned$ranked_table
)

gate_dir <- file.path(getwd(), "gate_results")
if (dir.exists(gate_dir)) unlink(gate_dir, recursive = TRUE, force = TRUE)
dir.create(gate_dir, recursive = TRUE, showWarnings = FALSE)

saveRDS(feature_selection, file.path(gate_dir, "feature_selection.rds"))


# ------------------------------------
# Rebuild matrices after pruning
# ------------------------------------
if (gate_threshold == 0) {
  num_features_keep <- num_features
  Z_features_keep   <- Z_features
} else {
  num_features_keep <- num_features_pruned$selected_features
  Z_features_keep   <- Z_features_pruned$selected_features
}

X_num_all_p <- as.matrix(dt[, ..num_features_keep])
Z_num_all_p <- as.matrix(dt[, ..Z_features_keep])

fit_num_p <- scale_fit(X_num_all_p[tr_rows, , drop = FALSE])
fit_Z_p   <- scale_fit(Z_num_all_p[tr_rows, , drop = FALSE])

X_num_tr_p <- scale_apply(X_num_all_p[tr_rows, , drop = FALSE], fit_num_p)
X_num_va_p <- scale_apply(X_num_all_p[va_rows, , drop = FALSE], fit_num_p)
X_num_te_p <- scale_apply(X_num_all_p[te_rows, , drop = FALSE], fit_num_p)
X_num_fo_p <- scale_apply(X_num_all_p[fo_rows, , drop = FALSE], fit_num_p)

Z_num_tr_p <- scale_apply(Z_num_all_p[tr_rows, , drop = FALSE], fit_Z_p)
Z_num_va_p <- scale_apply(Z_num_all_p[va_rows, , drop = FALSE], fit_Z_p)
Z_num_te_p <- scale_apply(Z_num_all_p[te_rows, , drop = FALSE], fit_Z_p)
Z_num_fo_p <- scale_apply(Z_num_all_p[fo_rows, , drop = FALSE], fit_Z_p)

# ------------------------------------
# Pack pruned data
# ------------------------------------
train_data_p <- list(
  x_with = make_x(TRUE,  X_num_tr_p, Z_num_tr_p, Z_bin_tr, off_tr, party_tr, kommun_tr, interact_tr),
  x_wo   = make_x(FALSE, X_num_tr_p, Z_num_tr_p, Z_bin_tr, off_tr, party_tr, kommun_tr, interact_tr),
  y = y_tr
)

val_data_p <- list(
  x_with = make_x(TRUE,  X_num_va_p, Z_num_va_p, Z_bin_va, off_va, party_va, kommun_va, interact_va),
  x_wo   = make_x(FALSE, X_num_va_p, Z_num_va_p, Z_bin_va, off_va, party_va, kommun_va, interact_va),
  y = y_va
)

test_data_p <- list(
  x_with = make_x(TRUE,  X_num_te_p, Z_num_te_p, Z_bin_te, off_te, party_te, kommun_te, interact_te),
  x_wo   = make_x(FALSE, X_num_te_p, Z_num_te_p, Z_bin_te, off_te, party_te, kommun_te, interact_te),
  y = y_te
)

fo_data_p <- list(
  x_with = make_x(TRUE,  X_num_fo_p, Z_num_fo_p, Z_bin_fo, off_fo, party_fo, kommun_fo, interact_fo),
  x_wo   = make_x(FALSE, X_num_fo_p, Z_num_fo_p, Z_bin_fo, off_fo, party_fo, kommun_fo, interact_fo),
  y = y_fo
)

# ------------------------------------
# Combine TRAIN + VAL after pruning
# ------------------------------------
X_num_trva_p <- rbind(X_num_tr_p, X_num_va_p)
Z_num_trva_p <- rbind(Z_num_tr_p, Z_num_va_p)
Z_bin_trva_p <- rbind(Z_bin_tr, Z_bin_va)

off_trva      <- rbind(off_tr, off_va)
party_trva    <- rbind(party_tr, party_va)
kommun_trva   <- rbind(kommun_tr, kommun_va)
interact_trva <- rbind(interact_tr, interact_va)

y_trva <- c(y_tr, y_va)

trva_data_p <- list(
  x_with = make_x(TRUE,  X_num_trva_p, Z_num_trva_p, Z_bin_trva_p, off_trva, party_trva, kommun_trva, interact_trva),
  x_wo   = make_x(FALSE, X_num_trva_p, Z_num_trva_p, Z_bin_trva_p, off_trva, party_trva, kommun_trva, interact_trva),
  y = y_trva
)

# ------------------------------------
# Refit pruned model on TRAIN + VAL
# ------------------------------------
with_inter_final <- best$emb_inter > 0L
trva_x_p <- if (with_inter_final) trva_data_p$x_with else trva_data_p$x_wo
te_x_p   <- if (with_inter_final) test_data_p$x_with else test_data_p$x_wo

k_clear_session()
set_global_seed(2026)

final_model_p <- build_zinb_nn(
  lambda_en_cnt = prune_lambda_cnt,
  alpha_en_cnt  = 0.5,
  lambda_en_zi  = prune_lambda_zi,
  alpha_en_zi   = 0.5,

  num_features   = num_features_keep,
  Z_features     = Z_features_keep,
  Z_bin_features = Z_bin_features,

  n_party    = n_party,
  n_kommun   = n_kommun,
  n_interact = n_interact,

  emb_dim_party    = best$emb_party,
  emb_dim_kommun   = best$emb_kommun,
  emb_dim_interact = best$emb_inter,

  lr = lr,
  seed = 2026,
  w_zero = best$w_zero,

  use_gelu = use_gelu,
  use_softplus_mu = use_softplus_mu,

  init_log_mu_bias = b_mu0,
  init_logit_pi_bias = b_pi0,

  alpha_mode = alpha_mode,
  alpha_clip = alpha_clip,
  alpha_group_use = alpha_group_use,
  alpha_group_l2 = alpha_group_l2,
  alpha_global_init_log = alpha_global_init_log
)

history_final_p <- final_model_p %>% fit(
  x = trva_x_p,
  y = trva_data_p$y,
  epochs = 200,
  batch_size = 512,
  verbose = 2,
  shuffle = TRUE,
  workers = 1,
  use_multiprocessing = FALSE,
  callbacks = list(
    callback_early_stopping(
      monitor = "loss",
      patience = 15,
      restore_best_weights = TRUE
    ),
    callback_reduce_lr_on_plateau(
      monitor = "loss",
      factor = 0.5,
      patience = 4,
      min_lr = 1e-5
    )
  )
)

# ------------------------------------
# Save pruned train+val model
# ------------------------------------

#  save model Helper 
save_zinb_model <- function(model, model_dir,
                            best_params,
                            model_meta,
                            scalers,
                            index_maps,
                            splits,
                            training_hparams,
                            history = NULL,
                            with_inter_final,
                            training_data, testing_data=NULL
                          ) {

  # Always delete & recreate dir
  if (dir.exists(model_dir)) unlink(model_dir, recursive = TRUE)
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

  save_model_weights_hdf5(model, file.path(model_dir, "weights.h5"))
  saveRDS(best_params, file.path(model_dir, "best_hyperparams.rds"))
  saveRDS(model_meta, file.path(model_dir, "feature_meta.rds"))
  saveRDS(scalers, file.path(model_dir, "scalers.rds"))
  saveRDS(index_maps, file.path(model_dir, "index_maps.rds"))
  saveRDS(splits, file.path(model_dir, "splits.rds"))
  saveRDS(training_hparams, file.path(model_dir, "training_hparams.rds"))
  saveRDS(training_data, file.path(model_dir, "training_data.rds"))
  saveRDS(testing_data, file.path(model_dir, "testing_data.rds"))
  
  if (!is.null(history)) {
    message("Saving training history...")
    saveRDS(history, file.path(model_dir, "training_history.rds"))
  }
  saveRDS(list(with_inter_final = with_inter_final),
          file.path(model_dir, "prediction_meta.rds"))

}

# what to save
model_meta_p <- list(
  num_features = num_features_keep,
  Z_features = Z_features_keep,
  Z_bin_features = Z_bin_features,
  n_party = n_party,
  n_kommun = n_kommun,
  n_interact = n_interact
)

scalers_p <- list(
  fit_num = fit_num_p,
  fit_Z = fit_Z_p
)

index_maps <- list(
  party_map = unique(dt[, .(pty_code, party_factor)]),
  kommun_map = unique(dt[, .(lans_kommun_kod, kommun_factor)]),
  interact_map = unique(dt[, .(party_kommun, party_kommun_factor)])
)

splits <- list(
  tr_rows = tr_rows,
  va_rows = va_rows,
  te_rows = te_rows,
  fo_rows = fo_rows
)

training_hparams_p <- list(
  lr = lr,
  use_gelu = use_gelu,
  use_softplus_mu = use_softplus_mu,
  init_log_mu_bias = b_mu0,
  init_logit_pi_bias = b_pi0,
  alpha_mode = alpha_mode,
  alpha_clip = alpha_clip,
  alpha_group_use = alpha_group_use,
  alpha_group_l2 = alpha_group_l2,
  alpha_global_init_log = alpha_global_init_log,
  gate_threshold = gate_threshold
)

# save
save_zinb_model(
  model = final_model_p,
  model_dir = "trva_mod",
  best_params = best,
  model_meta = model_meta_p,
  scalers = scalers_p,
  index_maps = index_maps,
  splits = splits,
  training_hparams = training_hparams_p,
  history = history_final_p,
  with_inter_final = with_inter_final,
  training_data = list(x = trva_x_p, y = trva_data_p$y),
  testing_data = list(x = te_x_p, y = test_data_p$y)
)



#######################################################################################
# ------------ Inference by trva_pruned
#######################################################################################
source(file.path(getwd(), "functions_zinb_gate_2_prune.R"))

# -------------------------------------------------------
#  trva pruned model predictions
# -------------------------------------------------------
loaded <- load_zinb_model(file.path(getwd(), "trva_mod"))

model <- loaded$model
scalers <- loaded$scalers
meta <- loaded$meta
index_maps <- loaded$index_maps
with_inter_final <- loaded$with_inter_final
training_hparams <- loaded$training_hparams
training_data <- loaded$training_data
testing_data <- loaded$testing_data
best_hyperparams <- loaded$best_hyperparams

# sanity checks
cat("Model expects", length(model$inputs), "input tensors\n")
cat("Training data has", length(training_data$x), "input tensors\n")
cat("Testing data has", length(testing_data$x), "input tensors\n")
stopifnot(length(model$inputs) == length(training_data$x))
stopifnot(length(model$inputs) == length(testing_data$x))

# -------- predictions
infer_dir <- file.path(getwd(), "trva_infer")
if (dir.exists(infer_dir)) unlink(infer_dir, recursive = TRUE)
dir.create(infer_dir, recursive = TRUE, showWarnings = FALSE)

trva_plot <- predict_zinb_accuracy(
  model = model,
  x_data = training_data$x,
  y_true = training_data$y,
  use_softplus_mu = training_hparams$use_softplus_mu
)
trva_plot$set="1. Train+Val"

trva_row_id <- c(which(tr_rows), which(va_rows))

trva_diag_full <- cbind(
  dt[trva_row_id, .(
    time,
    lans_kommun_kod,
    kommun_name,
    pty_code,
    pty_short,
    votes,
    lag_votes,
    tot_elig_voter
  )],
  as.data.table(trva_plot)
)
fwrite(trva_diag_full, file.path(infer_dir, "trva_pred.csv"))

te_plot <- predict_zinb_accuracy(
  model = model,
  x_data = testing_data$x,
  y_true = testing_data$y,
  use_softplus_mu = training_hparams$use_softplus_mu
)
te_plot$set="2. Test"

te_row_id <- c(which(te_rows))

te_diag_full <- cbind(
  dt[te_row_id, .(
    time,
    lans_kommun_kod,
    kommun_name,
    pty_code,
    pty_short,
    votes,
    lag_votes,
    tot_elig_voter
  )],
  as.data.table(te_plot)
)
fwrite(te_diag_full, file.path(infer_dir, "te_pred.csv"))


#######################################################################################
# explain by trva pruned model
#######################################################################################
explain_dir <- file.path(getwd(), "trva_explain")
if (dir.exists(explain_dir)) unlink(explain_dir, recursive = TRUE)
dir.create(explain_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------ gating values
f <- list.files(file.path(getwd(), "gate_results"), full.names=T)
temp <- readRDS(f) 

gates_zero <- temp$gates_zero |> as.data.table()
gates_cnt <- temp$gates_count |> as.data.table()

# ------------------------ ablation
# ---- helper x num feature
ablate_numeric_feature <- function(model, x_data, feature_name,
                                   num_features, use_softplus_mu, y_true) {

  x0 <- x_data
  k <- match(feature_name, num_features)
  if (is.na(k)) stop("Feature not found: ", feature_name)

  # Zero out (standard practice in ablation)
  x0[[1]][, k] <- 0

  out <- predict_zinb_accuracy(model, x = x0, y_true = y_true,
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

  out <- predict_zinb_accuracy(model, x = x0, y_true = y_true,
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

  out <- predict_zinb_accuracy(model, x = x0, y_true = y_true,
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
  base <- predict_zinb_accuracy(model,
                       x_data = x_data,
                       y_true = y_true,
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
  with_inter = loaded$with_inter_final,
  use_softplus_mu = training_hparams$use_softplus_mu
) |> as.data.table()
abla <- temp[, c("dr2", "dmae"):=lapply(.SD, round, 3), .SDcols = c("dr2", "dmae")]
fwrite(abla, file.path(explain_dir, "ablation.csv"))

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
fwrite(elas, file.path(explain_dir, "elas.csv"))


# ------------------------ combine gates, abla, elas
temp <- merge(abla, elas, by=c("feature", "type"), all=T)
selcol <- grep("feature|type|dmae|elastici", names(temp), value=T)
temp <- temp[, ..selcol]

temp_gates <- merge(gates_zero, gates_cnt, by=c("feature"), all=T, suffixes = c("_zero", "_cnt"))
selcol <- grep("_raw", names(temp_gates), value=T)
temp_gates <- temp_gates[, -..selcol]

combine <- merge(temp, temp_gates, by="feature", all=T)
combine <- combine[order(-dmae)]
print(combine)
fwrite(combine, file.path(explain_dir, "all_explain.csv"))


#######################################################################################
# -------- Production refit: trvate, then forecast 2026
#######################################################################################

# Historical rows used for final production training
hist_rows <- tr_rows | va_rows | te_rows

# ------------------------------------
# Refit scalers on all historical data
# ------------------------------------
X_num_all_prod <- as.matrix(dt[, ..num_features_keep])
Z_num_all_prod <- as.matrix(dt[, ..Z_features_keep])

fit_num_prod <- scale_fit(X_num_all_prod[hist_rows, , drop = FALSE])
fit_Z_prod   <- scale_fit(Z_num_all_prod[hist_rows, , drop = FALSE])

X_num_hist_p <- scale_apply(X_num_all_prod[hist_rows, , drop = FALSE], fit_num_prod)
X_num_fo_p   <- scale_apply(X_num_all_prod[fo_rows,   , drop = FALSE], fit_num_prod)

Z_num_hist_p <- scale_apply(Z_num_all_prod[hist_rows, , drop = FALSE], fit_Z_prod)
Z_num_fo_p   <- scale_apply(Z_num_all_prod[fo_rows,   , drop = FALSE], fit_Z_prod)

Z_bin_hist_p <- Z_bin_all[hist_rows, , drop = FALSE]
Z_bin_fo_p   <- Z_bin_all[fo_rows,   , drop = FALSE]

storage.mode(Z_bin_hist_p) <- "double"
storage.mode(Z_bin_fo_p)   <- "double"

off_hist      <- offset[hist_rows, , drop = FALSE]
party_hist    <- party_idx[hist_rows, , drop = FALSE]
kommun_hist   <- kommun_idx[hist_rows, , drop = FALSE]
interact_hist <- interact_idx[hist_rows, , drop = FALSE]

off_fo      <- offset[fo_rows, , drop = FALSE]
party_fo    <- party_idx[fo_rows, , drop = FALSE]
kommun_fo   <- kommun_idx[fo_rows, , drop = FALSE]
interact_fo <- interact_idx[fo_rows, , drop = FALSE]

y_hist <- y[hist_rows]
y_fo   <- y[fo_rows]

# ------------------------------------
# Production bias initialization
# bounded μ: μ = eligible_voters * sigmoid(eta)
# ------------------------------------
exp_off_hist <- as.numeric(exp(off_hist))

rate_hist <- mean(y_hist / pmax(1e-9, exp_off_hist), na.rm = TRUE)
rate_hist <- pmin(pmax(rate_hist, 1e-6), 1 - 1e-6)

b_mu0_prod <- qlogis(rate_hist)

p0_hist <- mean(y_hist == 0, na.rm = TRUE)
b_pi0_prod <- qlogis(pmin(pmax(p0_hist, 1e-6), 1 - 1e-6))

# ------------------------------------
# Pack production train and forecast data
# ------------------------------------
with_inter_final <- best$emb_inter > 0L

hist_data_p <- list(
  x_with = make_x(
    TRUE,
    X_num_hist_p, Z_num_hist_p, Z_bin_hist_p, off_hist,
    party_hist, kommun_hist, interact_hist
  ),
  x_wo = make_x(
    FALSE,
    X_num_hist_p, Z_num_hist_p, Z_bin_hist_p, off_hist,
    party_hist, kommun_hist, interact_hist
  ),
  y = y_hist
)

fo_data_p <- list(
  x_with = make_x(
    TRUE,
    X_num_fo_p, Z_num_fo_p, Z_bin_fo_p, off_fo,
    party_fo, kommun_fo, interact_fo
  ),
  x_wo = make_x(
    FALSE,
    X_num_fo_p, Z_num_fo_p, Z_bin_fo_p, off_fo,
    party_fo, kommun_fo, interact_fo
  ),
  y = y_fo
)

hist_x_p <- if (with_inter_final) hist_data_p$x_with else hist_data_p$x_wo
fo_x_p   <- if (with_inter_final) fo_data_p$x_with   else fo_data_p$x_wo

# ------------------------------------
# Refit final production model
# ------------------------------------
k_clear_session()
set_global_seed(2026)

prod_model_p <- build_zinb_nn(
  lambda_en_cnt = prune_lambda_cnt,
  alpha_en_cnt  = 0.5,
  lambda_en_zi  = prune_lambda_zi,
  alpha_en_zi   = 0.5,

  num_features   = num_features_keep,
  Z_features     = Z_features_keep,
  Z_bin_features = Z_bin_features,

  n_party    = n_party,
  n_kommun   = n_kommun,
  n_interact = n_interact,

  emb_dim_party    = best$emb_party,
  emb_dim_kommun   = best$emb_kommun,
  emb_dim_interact = best$emb_inter,

  lr = lr,
  seed = 2026,
  w_zero = best$w_zero,

  use_gelu = use_gelu,
  use_softplus_mu = use_softplus_mu,

  init_log_mu_bias = b_mu0_prod,
  init_logit_pi_bias = b_pi0_prod,

  alpha_mode = alpha_mode,
  alpha_clip = alpha_clip,
  alpha_group_use = alpha_group_use,
  alpha_group_l2 = alpha_group_l2,
  alpha_global_init_log = alpha_global_init_log
)

history_prod_p <- prod_model_p %>% fit(
  x = hist_x_p,
  y = hist_data_p$y,
  epochs = 200,
  batch_size = 512,
  verbose = 2,
  shuffle = TRUE,
  workers = 1,
  use_multiprocessing = FALSE,
  callbacks = list(
    callback_early_stopping(
      monitor = "loss",
      patience = 15,
      restore_best_weights = TRUE
    ),
    callback_reduce_lr_on_plateau(
      monitor = "loss",
      factor = 0.5,
      patience = 4,      
      min_lr = 1e-5
    )
  )
)

# what to save for production model
model_meta_prod <- list(
  num_features = num_features_keep,
  Z_features = Z_features_keep,
  Z_bin_features = Z_bin_features,
  n_party = n_party,
  n_kommun = n_kommun,
  n_interact = n_interact
)

scalers_prod <- list(
  fit_num = fit_num_prod,
  fit_Z = fit_Z_prod
)

splits_prod <- list(
  tr_rows = tr_rows,
  va_rows = va_rows,
  te_rows = te_rows,
  fo_rows = fo_rows,
  hist_rows = hist_rows
)

training_hparams_prod <- list(
  lr = lr,
  use_gelu = use_gelu,
  use_softplus_mu = use_softplus_mu,
  init_log_mu_bias = b_mu0_prod,
  init_logit_pi_bias = b_pi0_prod,
  alpha_mode = alpha_mode,
  alpha_clip = alpha_clip,
  alpha_group_use = alpha_group_use,
  alpha_group_l2 = alpha_group_l2,
  alpha_global_init_log = alpha_global_init_log,
  gate_threshold = gate_threshold,
  lambda_en_cnt = prune_lambda_cnt,
  lambda_en_zi = prune_lambda_zi
)

save_zinb_model(
  model = prod_model_p,
  model_dir = "trvate_mod",
  best_params = best,
  model_meta = model_meta_prod,
  scalers = scalers_prod,
  index_maps = index_maps,
  splits = splits_prod,
  training_hparams = training_hparams_prod,
  history = history_prod_p,
  with_inter_final = with_inter_final,
  training_data = list(x = hist_x_p, y = hist_data_p$y),
  testing_data = NULL
)


# ------------------------------------
# Forecast 
# ------------------------------------
# helper
predict_zinb_forecast <- function(model, x_data, use_softplus_mu = FALSE) {

  stopifnot(length(model$inputs) == length(x_data))

  pred_full <- predict(model, x_data, verbose = 0)
  params <- extract_params(pred_full, use_softplus_mu = use_softplus_mu)

  pred <- (1 - params$pi) * params$mu

  tibble::tibble(
    pred = pred,
    mu = params$mu,
    alpha = params$alpha,
    pi = params$pi
  )
}

# forecast
forecast_2026 <- predict_zinb_forecast(
  model = prod_model_p,
  x_data = fo_x_p,
  use_softplus_mu = use_softplus_mu
)

# results
forecast_out <- cbind(
  dt[fo_rows, .(
    time,
    lans_kommun_kod,
    kommun_name,
    pty_code,
    pty_name,
    pty_short,
    tot_elig_voter,
    offset_log
  )],
  as.data.table(forecast_2026)
)

forecast_dir <- file.path(getwd(), "trvate_infer")
if (dir.exists(forecast_dir)) unlink(forecast_dir, recursive = TRUE, force = TRUE)
dir.create(forecast_dir, recursive = TRUE, showWarnings = FALSE)

saveRDS(
  list(x = fo_x_p, y = fo_data_p$y),
  file.path(forecast_dir, "forecast_data.rds")
)

saveRDS(
  forecast_out,
  file.path(forecast_dir, "forecast_2026_party_kommun.rds")
)

fwrite(
  forecast_out,
  file.path(forecast_dir, "forecast_2026_party_kommun.csv")
)


# tune soft-gated model
#   ↓
# refit soft model on train+val
#   ↓
# extract gates
#   ↓
# hard prune features using gate > xx
#   ↓
# refit pruned model on train+val
#   ↓
# evaluate on test
#   ↓
# refit pruned model on train+val+test
#   ↓
# forecast 2026

