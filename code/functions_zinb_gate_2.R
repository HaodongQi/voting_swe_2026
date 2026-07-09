

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

  log_mu_no_offset <- cnt %>% layer_dense(
    1,
    kernel_initializer = initializer_glorot_uniform(seed+5L),
    bias_initializer   = initializer_constant(init_log_mu_bias)
  )
  log_mu_hat <- layer_add(list(log_mu_no_offset, inp_off))

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
    if ("interact"%in% alpha_group_use) group_inputs <- c(group_inputs, list(emb_inter))

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

# ------------------------------------
#  save model Helper 
# ------------------------------------
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

#######################################################################################
# ------------ Inference functions
#######################################################################################

# ------------------------------------
# load model helper
# ------------------------------------
load_zinb_model <- function(model_dir) {

  best <- readRDS(file.path(model_dir, "best_hyperparams.rds"))
  meta <- readRDS(file.path(model_dir, "feature_meta.rds"))
  scalers <- readRDS(file.path(model_dir, "scalers.rds"))
  index_maps <- readRDS(file.path(model_dir, "index_maps.rds"))
  splits <- readRDS(file.path(model_dir, "splits.rds"))
  training_hparams <-  readRDS(file.path(model_dir, "training_hparams.rds"))
  training_data <-  readRDS(file.path(model_dir, "training_data.rds"))
  testing_data <-  readRDS(file.path(model_dir, "testing_data.rds"))
  pred_meta <- readRDS(file.path(model_dir, "prediction_meta.rds"))
  with_inter_final <- pred_meta$with_inter_final

  model <- build_zinb_nn(
    lambda_en_cnt = best$lambda_cnt,
    alpha_en_cnt  = 0.5,
    lambda_en_zi  = best$lambda_zi,
    alpha_en_zi   = 0.5,
    num_features   = meta$num_features,
    Z_features     = meta$Z_features,
    Z_bin_features = meta$Z_bin_features,
    n_party = meta$n_party,
    n_kommun = meta$n_kommun,
    n_interact = meta$n_interact,
    emb_dim_party    = best$emb_party,
    emb_dim_kommun   = best$emb_kommun,
    emb_dim_interact = best$emb_inter,
    lr = training_hparams$lr,
    seed = 2026,
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

  model %>% load_model_weights_hdf5(file.path(model_dir, "weights.h5"))

  list(
    model = model,
    best_hyperparams = best,
    scalers = scalers,
    index_maps = index_maps,
    splits=splits,
    meta = meta,
    with_inter_final = with_inter_final,
    training_hparams=training_hparams,
    training_data = training_data,
    testing_data = testing_data
  )
}

# ------------------------------------
# predict_zinb helper
# ------------------------------------
predict_zinb <- function(model,
                         x_data, 
                         y_true,
                         set_name = "set",
                         use_softplus_mu = FALSE,
                         with_inter = NULL) {

  # 1. Determine whether the model expects interaction
  if (is.null(with_inter)) {
    n_expected <- length(model$inputs)
    if (n_expected == 7L)      with_inter <- TRUE
    else if (n_expected == 6L) with_inter <- FALSE
    else stop(sprintf("Model expects %d inputs; unsupported.", n_expected))
  }

  # 3. Run model prediction
  pred_full <- predict(model, x_data, verbose = 0)

  # 4. Extract ZINB parameters
  params <- extract_params(pred_full, use_softplus_mu = use_softplus_mu)
  mu     <- params$mu
  alpha  <- params$alpha
  pi     <- params$pi
  pred   <- (1 - pi) * mu                       # expected counts

  # 5. Optional evaluation
  auc <- r2 <- NA_real_
  if (!is.null(y_true)) {
    is_zero <- as.integer(y_true == 0)
    auc <- tryCatch(
      pROC::auc(pROC::roc(response = is_zero, predictor = pi)) |> as.numeric(),
      error = function(e) NA_real_
    )
    r2 <- tryCatch(
      cor(y_true, pred, use = "complete.obs")^2,
      error = function(e) NA_real_
    )
  }

  tibble::tibble(
    set = set_name,
    true = y_true,
    pred = pred,
    mu = mu,
    alpha = alpha,
    pi = pi,
    auc = auc,
    r2_corr = r2
  )
}

# -------------------------------------------------------
# extract params helpers
# -------------------------------------------------------
extract_params <- function(model_out, use_softplus_mu) {
  log_mu_hat    <- model_out[, 1]
  log_alpha_hat <- model_out[, 2]
  logit_pi      <- model_out[, 3]
  mu    <- if (use_softplus_mu) log1p(exp(log_mu_hat)) else exp(log_mu_hat)
  alpha <- log1p(exp(log_alpha_hat))
  pi    <- plogis(logit_pi)
  list(mu = mu, alpha = alpha, pi = pi)
}

