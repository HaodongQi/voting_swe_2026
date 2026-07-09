
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


# ----------------------------
# Elastic Net Gating Layer
# ----------------------------
ElNetGateLayer <- R6::R6Class(
  "ElNetGateLayer",
  inherit = KerasLayer,
  public = list(
    gate_vector = NULL, lambda_en = NULL, alpha_en = NULL,

    initialize = function(lambda_en = 5e-4, alpha_en = 0.5) {
      self$lambda_en <- lambda_en
      self$alpha_en  <- alpha_en
    },

    build = function(input_shape) {
      tf <- tensorflow::tf  # bind locally for reticulate callback safety
      num_features <- as.integer(input_shape[[2]])
      self$gate_vector <- self$add_weight(
        name = "gate_vector",
        shape = list(num_features),
        initializer = initializer_constant(0.0),  # you can try -1.0/-1.5 if you want sparser start
        trainable = TRUE
      )
    },

    call = function(x, mask = NULL) {
      tf <- tensorflow::tf

      gates <- tf$nn$sigmoid(self$gate_vector)

      en_loss <- self$lambda_en * (
        self$alpha_en * tf$reduce_sum(tf$abs(self$gate_vector)) +
          (1 - self$alpha_en) * tf$reduce_sum(tf$square(self$gate_vector))
      )

      self$add_loss(en_loss)

      x * gates
    }
  )
)

layer_elnet_gates <- function(object, lambda_en = 5e-4, alpha_en = 0.5, name = "gates") {
  create_layer(ElNetGateLayer, object,
    list(lambda_en = lambda_en, alpha_en = alpha_en, name = name)
  )
}

# ----------------------------
# ZINB Loss Function (clean, toggle μ link)
# ----------------------------
zinb_nll <- function(y_true, out, w_zero, use_softplus_mu = TRUE) {

  tf <- tensorflow::tf
  k  <- keras::backend()

  y_true <- k$cast(y_true, "float32")

  log_mu_hat    <- out[, 1, drop = FALSE]
  log_alpha_hat <- out[, 2, drop = FALSE]
  logit_pi      <- out[, 3, drop = FALSE]

  # μ link: exp OR softplus (choose ONE place to apply it → here)
  mu    <- if (use_softplus_mu) k$softplus(log_mu_hat) else k$exp(log_mu_hat)
  alpha <- k$softplus(log_alpha_hat)  # dispersion must be > 0
  pi    <- k$sigmoid(logit_pi)

  eps <- 1e-8
  inv_alpha <- 1 / (alpha + eps)

  # NB(μ, α) zero probability component
  log_p0_nb <- inv_alpha * tf$math$log(1 / (1 + alpha * mu + eps))

  # NB log-prob for y
  log_p_nb <- (
    tf$math$lgamma(y_true + inv_alpha) -
    tf$math$lgamma(inv_alpha) -
    tf$math$lgamma(y_true + 1) +
    inv_alpha * tf$math$log(1 / (1 + alpha * mu + eps)) +
    y_true * tf$math$log((alpha * mu + eps) / (1 + alpha * mu + eps))
  )

  is_zero     <- k$cast(k$equal(y_true, 0), "float32")
  log_mix0    <- tf$math$log(pi + (1 - pi) * tf$math$exp(log_p0_nb) + eps)
  log_mix_pos <- tf$math$log(1 - pi + eps) + log_p_nb

  # Weighted zero term
  loss <- -(w_zero * is_zero * log_mix0 + (1 - is_zero) * log_mix_pos)
  k$mean(loss)
}

# -------------------------------------------------------
# global alpha layer
# -------------------------------------------------------
AlphaGlobalLayer <- R6::R6Class(
  "AlphaGlobalLayer",
  inherit = KerasLayer,

  public = list(
    alpha_log = NULL,
    init_log_alpha = NULL,

    initialize = function(init_log_alpha = 0.0) {
      self$init_log_alpha <- init_log_alpha
    },

    build = function(input_shape) {
      tf <- tensorflow::tf
      self$alpha_log <- self$add_weight(
        name = "log_alpha_global",
        shape = list(),   # <-- scalar
        dtype = "float32",
        initializer = initializer_constant(self$init_log_alpha),
        trainable = TRUE
      )
    },
    
  call = function(inputs, mask = NULL) {
    tf <- tensorflow::tf
    batch_size <- tf$shape(inputs)[1]
    a2 <- tf$reshape(self$alpha_log, shape = c(1L,1L))
    multiples <- tf$stack(list(batch_size, tf$constant(1L, dtype = tf$int32)))
    tf$tile(a2, multiples)
  }

  )
)

layer_alpha_global <- function(object, init_log_alpha = 0.0, name=NULL) {
  create_layer(
    AlphaGlobalLayer, object,
    list(init_log_alpha = init_log_alpha, name=name)
  )
}

# -------------------------------------------------------
# build_zinb_nn(): two gates + toggles + stable defaults
# -------------------------------------------------------
build_zinb_nn <- function(
  lambda_en_cnt, alpha_en_cnt,
  lambda_en_zi,  alpha_en_zi,

  num_features, Z_features, Z_bin_features,
  n_party, n_kommun, n_interact,
  emb_dim_party, emb_dim_kommun, emb_dim_interact,

  lr = 1e-3, seed = 2026,
  w_zero = 1,
  use_gelu = TRUE,
  use_softplus_mu = TRUE,
  init_log_mu_bias = 0.0,
  init_logit_pi_bias = 0.0,

  # ---------- DISPERSION TOGGLES ----------
  alpha_mode = c("row","group","global"),
  alpha_clip = c(-5, 5),
  alpha_group_use = c("party","kommun"),
  alpha_group_l2  = 1e-6,
  alpha_global_init_log = log(0.1)
) {

  alpha_mode <- match.arg(alpha_mode)
  tf <- tensorflow::tf

  # -----------------------------------------------------
  # INPUTS
  # -----------------------------------------------------
  inp_num_cnt <- layer_input(shape=length(num_features),   dtype="float32", name="num_input_count")
  inp_num_zi  <- layer_input(shape=length(Z_features),     dtype="float32", name="num_input_zero")
  inp_Z       <- layer_input(shape=length(Z_bin_features), dtype="float32", name="ZI_input")
  inp_off     <- layer_input(shape=1, dtype="float32",     name="offset_input")
  inp_party   <- layer_input(shape=1, dtype="int32",       name="party_input")
  inp_kommun  <- layer_input(shape=1, dtype="int32",       name="kommun_input")
  inp_inter <- if (emb_dim_interact > 0) {
    layer_input(shape=1, dtype="int32", name="interact_input")
  } else NULL

  # -----------------------------------------------------
  # EMBEDDINGS
  # -----------------------------------------------------
  emb_kommun <- inp_kommun %>%
    layer_embedding(n_kommun+1, emb_dim_kommun,
                    embeddings_initializer = initializer_constant(0.0),
                    embeddings_regularizer = regularizer_l2(1e-6)) %>%
    layer_flatten()

  emb_party <- inp_party %>%
    layer_embedding(n_party+1, emb_dim_party,
                    embeddings_initializer = initializer_constant(0.0),
                    embeddings_regularizer = regularizer_l2(1e-6)) %>%
    layer_flatten()

  if (!is.null(inp_inter)) {
    emb_inter <- inp_inter %>%
      layer_embedding(n_interact+1, emb_dim_interact,
                      embeddings_initializer = initializer_constant(0.0),
                      embeddings_regularizer = regularizer_l2(5e-6)) %>%
      layer_flatten()
  }

  # -----------------------------------------------------
  # EN GATES
  # -----------------------------------------------------
  X_gated_cnt <- inp_num_cnt %>% layer_elnet_gates(lambda_en_cnt, alpha_en_cnt, name="gates_count")
  Z_gated_zi  <- inp_num_zi  %>% layer_elnet_gates(lambda_en_zi,  alpha_en_zi,  name="gates_zero")

  act <- if (use_gelu) tf$nn$gelu else tf$nn$relu

  # -----------------------------------------------------
  # COUNT TOWER (μ)
  # -----------------------------------------------------
  cnt_inputs <- list(X_gated_cnt, emb_party, emb_kommun)
  if (!is.null(inp_inter)) {
    cnt_inputs <- c(cnt_inputs, list(emb_inter))
  }

  cnt <- layer_concatenate(cnt_inputs) %>%
    layer_dense(64, activation=act, kernel_initializer=initializer_glorot_uniform(seed+3L)) %>%
    layer_dropout(0.1) %>%
    layer_dense(32, activation=act, kernel_initializer=initializer_glorot_uniform(seed+4L))

  # eta_vote_share is the logit of party vote share.
  # vote_share = sigmoid(eta_vote_share)
  # μ = eligible_voters * vote_share
  # log(μ) = offset_log + log(sigmoid(eta_vote_share))
  # This guarantees μ < eligible_voters for each party row.
  eta_vote_share <- cnt %>% 
    layer_dense(
      1,
      kernel_initializer = initializer_glorot_uniform(seed + 5L),
      bias_initializer   = initializer_constant(init_log_mu_bias),
      name = "eta_vote_share"
    )

  log_mu_hat <- layer_lambda(
    list(eta_vote_share, inp_off),
    function(z) {
      tf <- tensorflow::tf

      eta <- z[[1]]
      off <- z[[2]]
      eps <- 1e-8

      off + tf$math$log(tf$nn$sigmoid(eta) + eps)
    },
    name = "bounded_log_mu"
  )


  # -----------------------------------------------------
  # DISPERSION α HEAD (toggle)
  # -----------------------------------------------------
  if (alpha_mode == "row") {

    log_alpha_hat <- cnt %>%
      layer_dense(
        1,
        kernel_initializer = initializer_glorot_uniform(seed+6L),
        kernel_regularizer = regularizer_l2(1e-6)
      ) %>%
      layer_lambda(function(x) tf$clip_by_value(x, alpha_clip[1], alpha_clip[2]),
                   name="alpha_row_clip")

  } else if (alpha_mode == "group") {

    # Which embeddings to include?

    group_inputs <- list()
    if ("party"   %in% alpha_group_use) group_inputs <- c(group_inputs, list(emb_party))
    if ("kommun"  %in% alpha_group_use) group_inputs <- c(group_inputs, list(emb_kommun))
    if (!is.null(inp_inter) && "interact" %in% alpha_group_use) group_inputs <- c(group_inputs, list(emb_inter))

    if (length(group_inputs) == 0L) {
      stop("alpha_mode='group' but alpha_group_use is empty; choose at least one of {'party','kommun','interact'}.")
    }
    grp_in <- if (length(group_inputs) == 1L) group_inputs[[1]] else layer_concatenate(group_inputs)

    log_alpha_hat <- grp_in %>%
      layer_dense(1, activation = NULL,
                  kernel_regularizer = regularizer_l2(alpha_group_l2)) %>%
      layer_lambda(function(x) tf$clip_by_value(x, alpha_clip[1], alpha_clip[2]),
                  name = "alpha_group_clip")

  } else if (alpha_mode == "global") {

    log_alpha_hat <- inp_off %>% 
      layer_alpha_global(init_log_alpha = alpha_global_init_log,
                         name="alpha_global_layer")
  }

  # -----------------------------------------------------
  # ZERO‑INFLATION (π) TOWER
  # -----------------------------------------------------
  zi_inputs  <- list(inp_Z, Z_gated_zi, emb_party, emb_kommun)
  if (!is.null(inp_inter)) {
    zi_inputs  <- c(zi_inputs,  list(emb_inter))
  }

  zi <- layer_concatenate(zi_inputs) %>%
    layer_dense(32, activation=act, kernel_initializer=initializer_glorot_uniform(seed+7L)) %>%
    layer_dropout(0.1) %>%
    layer_dense(16, activation=act, kernel_initializer=initializer_glorot_uniform(seed+8L))

  logit_pi <- zi %>% layer_dense(
    1,
    kernel_initializer = initializer_glorot_uniform(seed+9L),
    bias_initializer   = initializer_constant(init_logit_pi_bias)
  )

  # -----------------------------------------------------
  # OUTPUT
  # -----------------------------------------------------
  out <- layer_concatenate(list(log_mu_hat, log_alpha_hat, logit_pi))

  # MODEL inputs list
  inputs_list <- list(inp_num_cnt, inp_num_zi, inp_Z, inp_off, inp_party, inp_kommun)
  if (!is.null(inp_inter)) inputs_list <- c(inputs_list, list(inp_inter))

  model <- keras_model(inputs = inputs_list, outputs = out)

  model %>% compile(
    optimizer = optimizer_adam(lr),
    loss = function(y_true, y_pred)
      zinb_nll(y_true, y_pred,
               w_zero = w_zero,
               use_softplus_mu = use_softplus_mu)
  )

  model
}

# =====================================================================
# ZINB Coarse-to-Fine Tuner (safe + consolidated)
# - two EN gates (μ and π) tuned independently
# - robust to TF session clears; never returns invalid model handles
# - easy wrapper to rebuild & refit the final model from best params
# =====================================================================

# -------------------------------
# Deterministic seeding helper
# -------------------------------
set_global_seed <- function(seed = 2026) {
  set.seed(seed)
  if (reticulate::py_module_available("numpy")) {
    reticulate::import("numpy")$random$seed(as.integer(seed))
  }
  tensorflow::tf$random$set_seed(as.integer(seed))
}

# -------------------------------
# Log-uniform sampler in [lo, hi]
# -------------------------------
.sample_log_uniform <- function(lo, hi, n) {
  if (length(lo) != 1 || length(hi) != 1 || lo <= 0 || hi <= 0 || lo >= hi)
    stop("Provide 0 < lo < hi for log-uniform sampling.")
  lam <- exp(runif(n, min = log(lo), max = log(hi)))
  as.numeric(lam)
}

# ------------------------------------------------------------
# Build a small local grid around a center on a log scale
# ------------------------------------------------------------
.local_log_grid <- function(center, factors, lo, hi) {
  g <- as.numeric(center) * factors
  g <- g[g > 0]
  g <- pmax(lo, pmin(hi, g))
  sort(unique(g))
}

# -----------------------------------------------------------------
# One trial: build -> fit -> return val_loss  (NO model returned)
# - Robust to builder/fit errors (returns Inf on error)
# - Always clears session between trials
# -----------------------------------------------------------------
.fit_once_two_gates <- function(
  lambda_cnt, lambda_zi,
  alpha_en_cnt, alpha_en_zi,
  train_data, val_data,
  num_features, Z_features, Z_bin_features,
  n_party, n_kommun, n_interact,
  emb_dim_party, emb_dim_kommun, emb_dim_interact,
  lr, seed,
  use_gelu, use_softplus_mu, w_zero,
  init_log_mu_bias, init_logit_pi_bias,
  alpha_mode, alpha_clip, alpha_group_use, alpha_group_l2, alpha_global_init_log,
  epochs, batch_size, patience = 5, verbose_fit = 0
) {
  k_clear_session()
  set_global_seed(seed)

  # defensive cap to avoid degenerate batches
  if (!is.null(train_data$y)) {
    batch_size <- min(batch_size, max(8L, nrow(matrix(train_data$y))))
  }

  # --------- choose x-lists based on emb_dim_interact ---------
  with_inter <- emb_dim_interact > 0L
  # Expect caller to provide either:
  #   train_data$x_with / train_data$x_wo    and   val_data$x_with / val_data$x_wo
  # or the old single-lists train_data$x / val_data$x
  x_tr <- if (!is.null(train_data$x_with) && !is.null(train_data$x_wo)) {
            if (with_inter) train_data$x_with else train_data$x_wo
          } else train_data$x
  x_va <- if (!is.null(val_data$x_with) && !is.null(val_data$x_wo)) {
            if (with_inter) val_data$x_with else val_data$x_wo
          } else val_data$x

  # Build
  model <- tryCatch({
    build_zinb_nn(
      lambda_en_cnt = lambda_cnt,
      alpha_en_cnt  = alpha_en_cnt,
      lambda_en_zi  = lambda_zi,
      alpha_en_zi   = alpha_en_zi,

      num_features   = num_features,
      Z_features     = Z_features,
      Z_bin_features = Z_bin_features,

      n_party    = n_party,
      n_kommun   = n_kommun,
      n_interact = n_interact,

      emb_dim_party    = emb_dim_party,
      emb_dim_kommun   = emb_dim_kommun,
      emb_dim_interact = emb_dim_interact,

      lr = lr, seed = seed,

      # toggles / knobs
      use_gelu         = use_gelu,
      use_softplus_mu  = use_softplus_mu,
      w_zero           = w_zero,
      init_log_mu_bias   = init_log_mu_bias,
      init_logit_pi_bias = init_logit_pi_bias,
      alpha_mode = alpha_mode,
      alpha_clip = alpha_clip,
      alpha_group_use = alpha_group_use,
      alpha_group_l2  = alpha_group_l2,
      alpha_global_init_log = alpha_global_init_log
    )
  }, error = function(e) {
    message("build_zinb_nn failed: ", conditionMessage(e))
    return(NULL)
  })

  if (is.null(model)) {
    return(list(val_loss = Inf))
  }

  # Fit
  history <- tryCatch({
    if (!is.null(val_data$y) && length(val_data$y)) {
      model %>% fit(
        x = x_tr, y = train_data$y,
        validation_data = list(x_va, val_data$y),
        epochs = epochs, batch_size = batch_size,
        shuffle = FALSE,
        workers = 1, use_multiprocessing = FALSE,
        verbose = verbose_fit,
        callbacks = list(
          callback_early_stopping(monitor = "val_loss",
                                  patience = patience,
                                  restore_best_weights = TRUE)
        )
      )
    } else {
      # Fallback: no val set — train only (not recommended for tuning)
      model %>% fit(
        x = x_tr, y = train_data$y,
        epochs = epochs, batch_size = batch_size,
        shuffle = FALSE,
        workers = 1, use_multiprocessing = FALSE,
        verbose = verbose_fit
      )
    }
  }, error = function(e) {
    message("fit failed: ", conditionMessage(e))
    return(NULL)
  })

  if (is.null(history)) {
    return(list(val_loss = Inf))
  }

  val_loss <- tryCatch({
    if (!is.null(val_data$y) && length(val_data$y)) {
      suppressWarnings(min(history$metrics$val_loss))
    } else {
      # No validation → use last batch train loss (weak proxy)
      suppressWarnings(tail(history$metrics$loss, 1))
    }
  }, error = function(e) Inf)

  if (!is.finite(val_loss)) val_loss <- Inf
  list(val_loss = val_loss)
}

# -----------------------------------------------------------------
# Two-stage tuner:
#  - Stage 1: random over λ_cnt, λ_zi, w_zero, emb dims
#  - Stage 2: local refine λ around Top-K seeds (dims fixed)
# Returns ONLY metrics + best hyper-parameters (no model)
# -----------------------------------------------------------------
coarse_to_fine_two_gates <- function(
  # Stage 1 (coarse)
  cnt_range, zi_range, n_stage1,

  # Stage 2 (refine)
  top_k, factors, n_max_stage2,

  # Model/training settings
  alpha_en_cnt = 0.5, alpha_en_zi = 0.5,
  train_data, val_data,
  num_features, Z_features, Z_bin_features,
  n_party, n_kommun, n_interact,

  # search grids
  w_zero_grid, emb_party_grid, emb_kommun_grid, emb_inter_grid,

  # current dims (not used in search; just for signature parity)
  emb_dim_party, emb_dim_kommun, emb_dim_interact,
  lr, seed,
  use_gelu, use_softplus_mu,
  init_log_mu_bias, init_logit_pi_bias,
  alpha_mode, alpha_clip, alpha_group_use, alpha_group_l2, alpha_global_init_log,

  batch_size = 512,
  epochs_stage1 = 40, patience_stage1 = 3,
  epochs_stage2 = 80, patience_stage2 = 6,
  verbose_fit = 0,
  allow_zero_interact = FALSE  # set TRUE only if your builder skips interact when dim=0
) {

  if (!allow_zero_interact && any(emb_inter_grid <= 0)) {
    warning("emb_inter_grid contains 0 but allow_zero_interact=FALSE. ",
            "Replacing non-positive values with 1 to avoid builder errors.")
    emb_inter_grid <- unique(pmax(1L, emb_inter_grid))
  }

  message(sprintf(
    "Stage 1: %d random trials over λ_cnt∈[%g,%g], λ_zi∈[%g,%g], |w_zero_grid|=%d, emb grids P/K/I=%d/%d/%d",
    n_stage1, cnt_range[1], cnt_range[2], zi_range[1], zi_range[2],
    length(w_zero_grid), length(emb_party_grid), length(emb_kommun_grid), length(emb_inter_grid)
  ))

  # Random samples for Stage 1
  lambda_cnt_trials <- .sample_log_uniform(cnt_range[1], cnt_range[2], n_stage1)
  lambda_zi_trials  <- .sample_log_uniform(zi_range[1],  zi_range[2],  n_stage1)
  w_zero_trials     <- sample(w_zero_grid,     size = n_stage1, replace = TRUE)
  emb_party_trials  <- sample(emb_party_grid,  size = n_stage1, replace = TRUE)
  emb_kommun_trials <- sample(emb_kommun_grid, size = n_stage1, replace = TRUE)
  emb_inter_trials  <- sample(emb_inter_grid,  size = n_stage1, replace = TRUE)

  s1 <- data.frame(
    lambda_cnt = lambda_cnt_trials,
    lambda_zi  = lambda_zi_trials,
    w_zero     = w_zero_trials,
    emb_party  = emb_party_trials,
    emb_kommun = emb_kommun_trials,
    emb_inter  = emb_inter_trials,
    val_loss   = NA_real_
  )

  best_loss <- Inf
  best_ix   <- NA_integer_

  # -------- Stage 1 loop --------
  for (i in seq_len(n_stage1)) {
    lam_cnt <- s1$lambda_cnt[i]; lam_zi <- s1$lambda_zi[i]
    wz <- s1$w_zero[i]; dp <- s1$emb_party[i]; dk <- s1$emb_kommun[i]; di <- s1$emb_inter[i]

    message(sprintf("Stage 1 %d/%d: λ_cnt=%.3g λ_zi=%.3g | w_zero=%.2f | emb(P/K/I)=%d/%d/%d",
                    i, n_stage1, lam_cnt, lam_zi, wz, dp, dk, di))

    res <- .fit_once_two_gates(
      lambda_cnt = lam_cnt, lambda_zi = lam_zi,
      alpha_en_cnt = alpha_en_cnt, alpha_en_zi = alpha_en_zi,
      train_data = train_data, val_data = val_data,
      num_features = num_features, Z_features = Z_features, Z_bin_features = Z_bin_features,
      n_party = n_party, n_kommun = n_kommun, n_interact = n_interact,
      emb_dim_party = dp, emb_dim_kommun = dk, emb_dim_interact = di,
      lr = lr, seed = seed,
      use_gelu = use_gelu, use_softplus_mu = use_softplus_mu, w_zero = wz,
      init_log_mu_bias = init_log_mu_bias, init_logit_pi_bias = init_logit_pi_bias,
      alpha_mode = alpha_mode, alpha_clip = alpha_clip, alpha_group_use = alpha_group_use,
      alpha_group_l2 = alpha_group_l2, alpha_global_init_log = alpha_global_init_log,
      epochs = epochs_stage1, batch_size = batch_size,
      patience = patience_stage1, verbose_fit = verbose_fit
    )

    s1$val_loss[i] <- res$val_loss
    if (res$val_loss < best_loss) { best_loss <- res$val_loss; best_ix <- i }
  }

  s1 <- s1[order(s1$val_loss), , drop = FALSE]
  top_k <- min(top_k, nrow(s1))
  s1_top <- s1[seq_len(top_k), , drop = FALSE]
  message("Top candidates (Stage 1):")
  print(s1_top)

  # -------- Stage 2 (local refine around Top-K) --------
  message("Stage 2: local refinement of λ around top candidates; holding w_zero & embeddings")
  s2_rows  <- list()
  s2_count <- 0L
  best2_loss <- best_loss
  best2_par  <- s1[best_ix, , drop = FALSE]

  for (k in seq_len(top_k)) {
    center_cnt <- s1_top$lambda_cnt[k]
    center_zi  <- s1_top$lambda_zi[k]
    wz_k       <- s1_top$w_zero[k]
    dp_k       <- s1_top$emb_party[k]
    dk_k       <- s1_top$emb_kommun[k]
    di_k       <- s1_top$emb_inter[k]

    grid_cnt <- .local_log_grid(center_cnt, factors, cnt_range[1], cnt_range[2])
    grid_zi  <- .local_log_grid(center_zi,  factors, zi_range[1],  zi_range[2])
    gk <- unique(expand.grid(lambda_cnt = grid_cnt, lambda_zi = grid_zi, KEEP.OUT.ATTRS = FALSE))

    for (j in seq_len(nrow(gk))) {
      if (s2_count >= n_max_stage2) break
      lam_cnt <- gk$lambda_cnt[j]; lam_zi <- gk$lambda_zi[j]
      s2_count <- s2_count + 1L

      message(sprintf("Stage 2 trial %d: λ_cnt=%.3g λ_zi=%.3g | w_zero=%.2f | emb=%d/%d/%d",
                      s2_count, lam_cnt, lam_zi, wz_k, dp_k, dk_k, di_k))

      res <- .fit_once_two_gates(
        lambda_cnt = lam_cnt, lambda_zi = lam_zi,
        alpha_en_cnt = alpha_en_cnt, alpha_en_zi = alpha_en_zi,
        train_data = train_data, val_data = val_data,
        num_features = num_features, Z_features = Z_features, Z_bin_features = Z_bin_features,
        n_party = n_party, n_kommun = n_kommun, n_interact = n_interact,
        emb_dim_party = dp_k, emb_dim_kommun = dk_k, emb_dim_interact = di_k,
        lr = lr, seed = seed,
        use_gelu = use_gelu, use_softplus_mu = use_softplus_mu, w_zero = wz_k,
        init_log_mu_bias = init_log_mu_bias, init_logit_pi_bias = init_logit_pi_bias,
        alpha_mode = alpha_mode, alpha_clip = alpha_clip, alpha_group_use = alpha_group_use,
        alpha_group_l2 = alpha_group_l2, alpha_global_init_log = alpha_global_init_log,
        epochs = epochs_stage2, batch_size = batch_size,
        patience = patience_stage2, verbose_fit = verbose_fit
      )

      s2_rows[[length(s2_rows)+1L]] <- data.frame(
        lambda_cnt = lam_cnt, lambda_zi = lam_zi,
        w_zero = wz_k, emb_party = dp_k, emb_kommun = dk_k, emb_inter = di_k,
        val_loss = res$val_loss
      )

      if (res$val_loss < best2_loss) {
        best2_loss <- res$val_loss
        best2_par <- data.frame(
          lambda_cnt = lam_cnt, lambda_zi = lam_zi,
          w_zero = wz_k, emb_party = dp_k, emb_kommun = dk_k, emb_inter = di_k,
          val_loss = res$val_loss
        )
      }
    }
    if (s2_count >= n_max_stage2) break
  }

  s2 <- if (length(s2_rows)) do.call(rbind, s2_rows) else
    data.frame(lambda_cnt = numeric(), lambda_zi = numeric(),
               w_zero = numeric(), emb_party = integer(),
               emb_kommun = integer(), emb_inter = integer(),
               val_loss = numeric())
  if (nrow(s2)) s2 <- s2[order(s2$val_loss), , drop = FALSE]

  # -------- save tuner results --------
  outdir <- file.path(getwd(), "tuner_results")
  if(dir.exists(outdir)){unlink(outdir, recursive = TRUE, force = TRUE)} 
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  

  # Save Stage 1, Stage 2 and best
  write.csv(s1, file.path(outdir, "stage1_trials.csv"), row.names = FALSE)
  write.csv(s2, file.path(outdir, "stage2_trials.csv"), row.names = FALSE)
  saveRDS(best2_par, file.path(outdir, "best_params.rds"))
  saveRDS(list(stage1 = s1, stage2 = s2, best = best2_par),
          file.path(outdir, "tuner_metadata.rds"))
  message("Tuning metadata saved to: ", outdir)

  # -------- output --------
  list(
    stage1_results = s1,
    stage2_results = s2,
    best = list(
      lambda_cnt = best2_par$lambda_cnt[1],
      lambda_zi  = best2_par$lambda_zi[1],
      w_zero     = best2_par$w_zero[1],
      emb_party  = best2_par$emb_party[1],
      emb_kommun = best2_par$emb_kommun[1],
      emb_inter  = best2_par$emb_inter[1],
      val_loss   = best2_par$val_loss[1]
    )
    # NOTE: no model returned; rebuild + refit using refit_best_model()
  )

}

# -----------------------------------------------------------------
# Rebuild + refit the final model from cf$best
# You can pass either:
#   - train_data,  val_data      (monitor val_loss)
#   - trainval_data, val_data    (fit on TRAIN+VAL; still monitor val)
# -----------------------------------------------------------------
refit_best_model <- function(
  best,
  train_data = NULL, val_data = NULL, trainval_data = NULL,
  num_features, Z_features, Z_bin_features,
  n_party, n_kommun, n_interact,
  lr, seed,
  use_gelu, use_softplus_mu,
  init_log_mu_bias, init_logit_pi_bias,
  alpha_mode, alpha_clip, alpha_group_use, alpha_group_l2, alpha_global_init_log,
  epochs = 200, batch_size = 512, patience = 10, verbose_fit = 2
) {
  # Select which dataset to fit on
  src <- 
    if (!is.null(trainval_data)) {
      "trainval"} else if (!is.null(train_data)) {
        "train"} else stop("Provide either train_data or trainval_data.")
  if (src == "trainval") { fit_in <- trainval_data } else { fit_in <- train_data }

  # ---- choose x list by best$emb_inter
  with_inter_final <- isTRUE(best$emb_inter > 0L)
  fit_x <- if (!is.null(fit_in$x_with) && !is.null(fit_in$x_wo)) {
    if (with_inter_final) fit_in$x_with else fit_in$x_wo
  } else fit_in$x
  fit_y <- fit_in$y

  val_x <- if (!is.null(val_data)) {
    if (!is.null(val_data$x_with) && !is.null(val_data$x_wo)) {
      if (with_inter_final) val_data$x_with else val_data$x_wo
    } else val_data$x
  } else NULL
  val_y <- if (!is.null(val_data)) val_data$y else NULL

  k_clear_session()
  set_global_seed(seed)

  model <- build_zinb_nn(
    lambda_en_cnt = best$lambda_cnt, alpha_en_cnt = 0.5,
    lambda_en_zi  = best$lambda_zi,  alpha_en_zi  = 0.5,
    num_features = num_features, Z_features = Z_features, Z_bin_features = Z_bin_features,
    n_party = n_party, n_kommun = n_kommun, n_interact = n_interact,
    emb_dim_party = best$emb_party, emb_dim_kommun = best$emb_kommun, emb_dim_interact = best$emb_inter,
    lr = lr, seed = seed,
    use_gelu = use_gelu, use_softplus_mu = use_softplus_mu,
    w_zero = best$w_zero,
    init_log_mu_bias = init_log_mu_bias, init_logit_pi_bias = init_logit_pi_bias,
    alpha_mode = alpha_mode, alpha_clip = alpha_clip,
    alpha_group_use = alpha_group_use, alpha_group_l2 = alpha_group_l2,
    alpha_global_init_log = alpha_global_init_log
  )

  if (!is.null(val_x) && length(val_y)) {
    history <- model %>% fit(
      x = fit_x, y = fit_y,
      validation_data = list(val_x, val_y),
      epochs = epochs, batch_size = min(batch_size, max(8L, length(fit_y))),
      shuffle = FALSE, workers = 1, use_multiprocessing = FALSE,
      verbose = verbose_fit,
      callbacks = list(
        callback_early_stopping(monitor="val_loss", patience=patience, restore_best_weights=TRUE),
        callback_reduce_lr_on_plateau(monitor="val_loss", factor=0.5, patience=4, min_lr=3e-5, cooldown=1)
      )
    )
  } else {
    warning("No validation data provided to refit_best_model(); training without validation.")
    history <- model %>% fit(
      x = fit_x, y = fit_y,
      epochs = epochs, batch_size = min(batch_size, max(8L, length(fit_y))),
      shuffle = FALSE, workers = 1, use_multiprocessing = FALSE,
      verbose = verbose_fit
    )
  }

  list(model = model, history = history)
}


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
      cooldown = 1
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
num_features_keep <- num_features_pruned$selected_features
Z_features_keep   <- Z_features_pruned$selected_features

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
      cooldown = 1
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


# -------------------------------------------------------
# Prediction helpers for loaded ZINB model
# -------------------------------------------------------
stable_softplus <- function(x) {
  ifelse(x > 30, x, log1p(exp(x)))
}

extract_params <- function(model_out, use_softplus_mu = FALSE) {
  log_mu_hat    <- model_out[, 1]
  log_alpha_hat <- model_out[, 2]
  logit_pi      <- model_out[, 3]

  mu <- if (use_softplus_mu) {
    stable_softplus(log_mu_hat)
  } else {
    exp(pmin(log_mu_hat, 30))
  }

  alpha <- stable_softplus(log_alpha_hat)
  pi    <- plogis(logit_pi)

  list(mu = mu, alpha = alpha, pi = pi)
}

predict_zinb_accuracy <- function(model,
                                  x_data,
                                  y_true,
                                  use_softplus_mu = FALSE) {

  # sanity check: correct number of tensors
  stopifnot(length(model$inputs) == length(x_data))

  pred_full <- predict(model, x_data, verbose = 0)
  params <- extract_params(pred_full, use_softplus_mu = use_softplus_mu)

  pred <- (1 - params$pi) * params$mu

  auc <- tryCatch(
    {
      is_zero <- as.integer(y_true == 0)
      pROC::auc(pROC::roc(response = is_zero, predictor = params$pi, quiet = TRUE)) |>
        as.numeric()
    },
    error = function(e) NA_real_
  )

  r2_corr <- tryCatch(
    cor(y_true, pred, use = "complete.obs")^2,
    error = function(e) NA_real_
  )

  mae <- mean(abs(y_true - pred), na.rm = TRUE)
  rmse <- sqrt(mean((y_true - pred)^2, na.rm = TRUE))

  tibble::tibble(
    true = y_true,
    pred = pred,
    mu = params$mu,
    alpha = params$alpha,
    pi = params$pi,
    auc = auc,
    r2_corr = r2_corr,
    mae = mae,
    rmse = rmse
  )
}

# ------------------------------------
# load model helper
# ------------------------------------
load_zinb_model <- function(model_dir,
                            seed = 2026,
                            load_training_data = TRUE,
                            load_testing_data = TRUE,
                            load_forecast_data = TRUE) {

  # -----------------------------
  # small safe RDS reader
  # -----------------------------
  read_rds_if_exists <- function(path, default = NULL) {
    if (file.exists(path)) readRDS(path) else default
  }

  # -----------------------------
  # required files
  # -----------------------------
  best <- readRDS(file.path(model_dir, "best_hyperparams.rds"))
  meta <- readRDS(file.path(model_dir, "feature_meta.rds"))
  scalers <- readRDS(file.path(model_dir, "scalers.rds"))
  index_maps <- readRDS(file.path(model_dir, "index_maps.rds"))
  splits <- readRDS(file.path(model_dir, "splits.rds"))
  training_hparams <- readRDS(file.path(model_dir, "training_hparams.rds"))
  pred_meta <- readRDS(file.path(model_dir, "prediction_meta.rds"))

  with_inter_final <- pred_meta$with_inter_final

  # -----------------------------
  # optional files
  # -----------------------------
  training_data <- if (load_training_data) {
    read_rds_if_exists(file.path(model_dir, "training_data.rds"))
  } else NULL

  testing_data <- if (load_testing_data) {
    read_rds_if_exists(file.path(model_dir, "testing_data.rds"))
  } else NULL

  forecast_data <- if (load_forecast_data) {
    read_rds_if_exists(file.path(model_dir, "forecast_data.rds"))
  } else NULL

  history <- read_rds_if_exists(file.path(model_dir, "training_history.rds"))
  gate_values <- read_rds_if_exists(file.path(model_dir, "gate_values.rds"))
  feature_selection <- read_rds_if_exists(file.path(model_dir, "feature_selection.rds"))

  # -----------------------------
  # rebuild architecture
  # -----------------------------
  k_clear_session()
  set_global_seed(seed)

  model <- build_zinb_nn(
    lambda_en_cnt = best$lambda_cnt,
    alpha_en_cnt  = 0.5,
    lambda_en_zi  = best$lambda_zi,
    alpha_en_zi   = 0.5,

    num_features   = meta$num_features,
    Z_features     = meta$Z_features,
    Z_bin_features = meta$Z_bin_features,

    n_party    = meta$n_party,
    n_kommun   = meta$n_kommun,
    n_interact = meta$n_interact,

    emb_dim_party    = best$emb_party,
    emb_dim_kommun   = best$emb_kommun,
    emb_dim_interact = best$emb_inter,

    lr = training_hparams$lr,
    seed = seed,
    w_zero = best$w_zero,

    use_gelu = training_hparams$use_gelu,
    use_softplus_mu = training_hparams$use_softplus_mu,

    init_log_mu_bias = training_hparams$init_log_mu_bias,
    init_logit_pi_bias = training_hparams$init_logit_pi_bias,

    alpha_mode = training_hparams$alpha_mode,
    alpha_clip = training_hparams$alpha_clip,
    alpha_group_use = training_hparams$alpha_group_use,
    alpha_group_l2  = training_hparams$alpha_group_l2,
    alpha_global_init_log = training_hparams$alpha_global_init_log
  )

  # -----------------------------
  # load weights
  # -----------------------------
  weights_path <- file.path(model_dir, "weights.h5")

  if (!file.exists(weights_path)) {
    stop("Cannot find weights file: ", weights_path)
  }

  model %>% load_model_weights_hdf5(weights_path)

  # -----------------------------
  # sanity checks
  # -----------------------------
  message("Loaded ZINB model from: ", normalizePath(model_dir))
  message("Model expects ", length(model$inputs), " input tensor(s).")
  message("with_inter_final = ", with_inter_final)

  if (!is.null(training_data)) {
    message("training_data$x has ", length(training_data$x), " input tensor(s).")
  }

  if (!is.null(testing_data)) {
    message("testing_data$x has ", length(testing_data$x), " input tensor(s).")
  }

  if (!is.null(forecast_data)) {
    message("forecast_data$x has ", length(forecast_data$x), " input tensor(s).")
  }

  # -----------------------------
  # return full bundle
  # -----------------------------
  list(
    model = model,

    best_hyperparams = best,
    meta = meta,
    scalers = scalers,
    index_maps = index_maps,
    splits = splits,
    training_hparams = training_hparams,

    with_inter_final = with_inter_final,

    training_data = training_data,
    testing_data = testing_data,
    forecast_data = forecast_data,

    history = history,
    gate_values = gate_values,
    feature_selection = feature_selection,

    model_dir = model_dir
  )
}

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
      cooldown = 1
    )
  )
)

# save
save_zinb_model(
  model = final_model_p,
  model_dir = "trvate_mod",
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

