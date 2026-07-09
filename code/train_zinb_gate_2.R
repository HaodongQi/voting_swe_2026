

# ------------------------------------
# config.
# ------------------------------------

# If running in RStudio, set working dir to script location (optional)
if (requireNamespace("rstudioapi", quietly = TRUE) &&
  rstudioapi::isAvailable()) {
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  }

pacman::p_load(tidyverse,data.table, keras, tfruns, reticulate)

# set virtual env.
use_virtualenv("~/v_environment/r-tf", required = TRUE)

# ---- Determinism toggles (set before loading tensorflow/keras) ----
Sys.setenv(PYTHONHASHSEED = "0")      # Python hash seed
Sys.setenv(TF_DETERMINISTIC_OPS = "1")# request deterministic TF ops (TF>=2.13)
set.seed(2026)                        # R's RNG

tensorflow::tf$random$set_seed(2026)

# ---- input enet and zinb functions ----
source("functions_zinb_gate_2.r")
source("preproc_zinb_nn.r")

# ------------------------------------
# training control
# ------------------------------------
# training stats for bias initialization (train split)
exp_off_tr <- as.numeric(exp(off_tr))
rate_tr <- if (length(y_tr)) mean(y_tr / pmax(1e-9, exp_off_tr)) else 1e-6
b_mu0 <- log(pmax(rate_tr, 1e-12))

p0_tr <- if (length(y_tr)) mean(y_tr == 0) else 0.5
b_pi0 <- qlogis(pmin(pmax(p0_tr, 1e-6), 1 - 1e-6))

# other control
use_gelu <- TRUE
use_softplus_mu <- FALSE
lr <- 8e-4

alpha_mode <- "group"          # kommun-specific α
alpha_clip <- c(-5, 5)
alpha_group_use <- c("kommun") # <-- kommun only
alpha_group_l2 <- 1e-6
alpha_global_init_log <- log(0.1)

# initial embedding dims (placeholders; tuner will pick best)
emb_dim_party <- 3L
emb_dim_kommun <- 8L
emb_dim_interact <- 1L   # set to 0 only if your builder skips interaction when 0


# ------------------------------------
# Coarse-to-fine tuning (two gates, no coupling)
# ------------------------------------
k_clear_session()
set_global_seed(2026)

cf <- coarse_to_fine_two_gates(
  cnt_range = c(1e-4, .99),
  zi_range  = c(5e-4, .99),
  n_stage1  = 100,

  top_k = 5,
  factors = c(0.25, 1, 2),
  n_max_stage2 = 200,

  alpha_en_cnt = 0.5, alpha_en_zi = 0.5,
  train_data = train_data, val_data = val_data,
  num_features = num_features, Z_features = Z_features, Z_bin_features = Z_bin_features,
  n_party = n_party, n_kommun = n_kommun, n_interact = n_interact,

  w_zero_grid     = c(1),
  emb_party_grid  = c(1, 2, 4, 8, 16),
  emb_kommun_grid = c(1, 2, 4, 8, 16),
  emb_inter_grid  = c(1),     # or c(0,1) only if builder supports zero-dim interaction

  emb_dim_party = emb_dim_party,
  emb_dim_kommun = emb_dim_kommun,
  emb_dim_interact = emb_dim_interact,

  lr = lr, seed = 2026,
  use_gelu = use_gelu,
  use_softplus_mu = use_softplus_mu,
  init_log_mu_bias = b_mu0,
  init_logit_pi_bias = b_pi0,

  alpha_mode = alpha_mode,
  alpha_clip = alpha_clip,
  alpha_group_use = alpha_group_use,
  alpha_group_l2  = alpha_group_l2,
  alpha_global_init_log = alpha_global_init_log,

  epochs_stage1 = 40, patience_stage1 = 3,
  epochs_stage2 = 80, patience_stage2 = 6,
  batch_size = 512, verbose_fit = 0,

  allow_zero_interact = TRUE  
)

message(sprintf("Best: λ_cnt=%.4g λ_zi=%.4g | w_zero=%.2f | emb=%d/%d/%d | val_loss=%.6f",
                cf$best$lambda_cnt, cf$best$lambda_zi, cf$best$w_zero,
                cf$best$emb_party, cf$best$emb_kommun, cf$best$emb_inter,
                cf$best$val_loss))

# ------------------------------------
#  best params
# ------------------------------------
f <- list.files(file.path(getwd(), "tuner_results"), recursive = T, full.names = T)
f <- grep("best_", f, ignore.case = T, value = T)
f <- readRDS(f)
best_lambda_cnt <- f$lambda_cnt
best_lambda_zi  <- f$lambda_zi
best_w_zero     <- f$w_zero
best_emb_party  <- f$emb_party
best_emb_kommun <- f$emb_kommun
best_emb_inter  <- f$emb_inter


# ------------------------------------
# Refit: Combine TRAIN + VAL  
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

k_clear_session()
set_global_seed(2026)

final_model <- build_zinb_nn(
  lambda_en_cnt = best_lambda_cnt,
  alpha_en_cnt  = 0.5,
  lambda_en_zi  = best_lambda_zi,
  alpha_en_zi   = 0.5,

  num_features   = num_features,
  Z_features     = Z_features,
  Z_bin_features = Z_bin_features,

  n_party = n_party, n_kommun = n_kommun, n_interact = n_interact,
  emb_dim_party    = best_emb_party,
  emb_dim_kommun   = best_emb_kommun,
  emb_dim_interact = best_emb_inter,

  lr = lr, seed = 2026,

  w_zero = best_w_zero,
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

# choose which input list to feed based on best_emb_inter
with_inter_final <- (best_emb_inter > 0L)

trva_x <- if (with_inter_final) {
  trva_data$x_with
} else {
  trva_data$x_wo
}

te_x <- if (with_inter_final) {
  test_data$x_with
} else {
  test_data$x_wo
}

# ---- refit with train+val loss
history_final <- final_model %>% fit(
  x = trva_x, y = trva_data$y,
  epochs = 200, batch_size = 512, verbose = 2,
  shuffle = FALSE, workers = 1, use_multiprocessing = FALSE,
  callbacks = list(
    callback_early_stopping(monitor="loss", patience=15, restore_best_weights=TRUE),
    callback_reduce_lr_on_plateau(monitor="loss", factor=0.5, patience=4, cooldown=1)
  )
)

# -------------- Save model --------------
model_meta <- list(
  num_features = num_features,
  Z_features = Z_features,
  Z_bin_features = Z_bin_features,
  n_party = n_party,
  n_kommun = n_kommun,
  n_interact = n_interact
)

scalers <- list(
  fit_num = fit_num,
  fit_Z   = fit_Z
)

index_maps <- list(
  party_map = dt[, .(pty_code, party_factor)],
  kommun_map = dt[, .(lans_kommun_kod, kommun_factor)],
  interact_map = dt[, .(party_kommun, party_kommun_factor)]
)

splits <- list(
  tr_rows = tr_rows,
  va_rows = va_rows
)

training_hparams <- list(
  lr = lr,
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

training_data <- list(x=trva_x, y=trva_data$y)

testing_data <- list(x=te_x, y=test_data$y)

save_zinb_model(
  model = final_model,
  model_dir = "trva_mod",
  best_params = f,
  model_meta = model_meta,
  scalers = scalers,
  index_maps = index_maps,
  splits = splits,
  training_hparams = training_hparams,
  history = history_final,
  with_inter_final = with_inter_final,
  training_data = training_data,
  testing_data = testing_data
)

# ------------------------------------
# Refit: Combine TRAIN + VAL  + TE. This is for forecasting 2026
# ------------------------------------
X_num_trvate <- rbind(X_num_tr, X_num_va, X_num_te)
Z_num_trvate <- rbind(Z_num_tr, Z_num_va, Z_num_te)
Z_bin_trvate <- rbind(Z_bin_tr, Z_bin_va, Z_bin_te)

off_trvate     <- rbind(off_tr, off_va, off_te)
party_trvate   <- rbind(party_tr, party_va, party_te)
kommun_trvate  <- rbind(kommun_tr, kommun_va, kommun_te)
interact_trvate <- rbind(interact_tr, interact_va, interact_te)

y_trvate <- c(y_tr, y_va, y_te)

trvate_data <- list(
  x_with = make_x(
    TRUE,
    X_num_trvate, Z_num_trvate, Z_bin_trvate, off_trvate,
    party_trvate, kommun_trvate, interact_trvate
  ),
  x_wo = make_x(
    FALSE,
    X_num_trvate, Z_num_trvate, Z_bin_trvate, off_trvate,
    party_trvate, kommun_trvate, interact_trvate
  ),
  y = y_trvate
)

k_clear_session()
set_global_seed(2026)

final_model <- build_zinb_nn(
  lambda_en_cnt = best_lambda_cnt,
  alpha_en_cnt  = 0.5,
  lambda_en_zi  = best_lambda_zi,
  alpha_en_zi   = 0.5,

  num_features   = num_features,
  Z_features     = Z_features,
  Z_bin_features = Z_bin_features,

  n_party = n_party, n_kommun = n_kommun, n_interact = n_interact,
  emb_dim_party    = best_emb_party,
  emb_dim_kommun   = best_emb_kommun,
  emb_dim_interact = best_emb_inter,

  lr = lr, seed = 2026,

  w_zero = best_w_zero,
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

# choose which input list to feed based on best_emb_inter
with_inter_final <- (best_emb_inter > 0L)

trvate_x <- if (with_inter_final) {
  trvate_data$x_with
} else {
  trvate_data$x_wo
}

te_x <- if (with_inter_final) {
  test_data$x_with
} else {
  test_data$x_wo
}

# ---- refit with train+val loss
history_final <- final_model %>% fit(
  x = trvate_x, y = trvate_data$y,
  epochs = 200, batch_size = 512, verbose = 2,
  shuffle = FALSE, workers = 1, use_multiprocessing = FALSE,
  callbacks = list(
    callback_early_stopping(monitor="loss", patience=15, restore_best_weights=TRUE),
    callback_reduce_lr_on_plateau(monitor="loss", factor=0.5, patience=4, cooldown=1)
  )
)

# -------------- Save model --------------
model_meta <- list(
  num_features = num_features,
  Z_features = Z_features,
  Z_bin_features = Z_bin_features,
  n_party = n_party,
  n_kommun = n_kommun,
  n_interact = n_interact
)

scalers <- list(
  fit_num = fit_num,
  fit_Z   = fit_Z
)

index_maps <- list(
  party_map = dt[, .(pty_code, party_factor)],
  kommun_map = dt[, .(lans_kommun_kod, kommun_factor)],
  interact_map = dt[, .(party_kommun, party_kommun_factor)]
)

splits <- list(
  tr_rows = tr_rows,
  va_rows = va_rows,
  te_rows = te_rows
)

training_hparams <- list(
  lr = lr,
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

training_data <- list(x=trvate_x, y=trvate_data$y)

testing_data <- list(x=te_x, y=test_data$y)

save_zinb_model(
  model = final_model,
  model_dir = "trvate_mod",
  best_params = f,
  model_meta = model_meta,
  scalers = scalers,
  index_maps = index_maps,
  splits = splits,
  training_hparams = training_hparams,
  history = history_final,
  with_inter_final = with_inter_final,
  training_data = training_data,
  testing_data = testing_data
)

