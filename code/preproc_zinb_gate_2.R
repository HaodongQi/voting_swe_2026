

# ------------------------------------
# config.
# ------------------------------------

# If running in RStudio, set working dir to script location (optional)
if (requireNamespace("rstudioapi", quietly = TRUE) &&
  rstudioapi::isAvailable()) {
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  }

pacman::p_load(tidyverse,data.table, keras, tfruns, reticulate)


# ------------------------------------
# pre process
# ------------------------------------
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