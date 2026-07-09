
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
source("preproc_zinb_gate_2.r")


# -------------------------------------------------------
# check trva prediciton
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

# flag if the final model used interaction
with_inter_final <- (best_hyperparams$emb_inter > 0L)

trva_plot <- predict_zinb(
  model,
  x_data = training_data$x, 
  y_true = training_data$y, set_name = "1. Train + Val", use_softplus_mu = training_hparams$use_softplus_mu,
  with_inter = with_inter_final                  # you can omit this; it auto-infers from model$inputs
)

te_plot <- predict_zinb(
  model,
  x_data = testing_data$x,
  y_true = testing_data$y, set_name = "2. Test", use_softplus_mu = training_hparams$use_softplus_mu,
  with_inter = with_inter_final
)

tr_va_te <- dplyr::bind_rows(trva_plot, te_plot)

# --- Through-origin regression per facet: pred ~ 0 + true (ORIGINAL scale) ---
reg_stats <- tr_va_te %>%
  group_by(set) %>%
  summarise(
    beta = {den <- sum(true^2); if (den <= .Machine$double.eps) NA_real_ else sum(true*pred)/den},
    r2_reg = {
      if (is.na(beta)) NA_real_ else {
        yfit <- beta*true
        sse  <- sum((pred - yfit)^2)
        sst0 <- sum(pred^2) # no-intercept convention when regressing pred on true
        if (sst0 <= .Machine$double.eps) NA_real_ else 1 - sse/sst0
      }
    },
    .groups = "drop"
  )

# --- Smooth fitted grid per facet: fitted(true) = beta * true, then plot on log1p axes ---
fit_lines <- tr_va_te %>%
  group_by(set) %>%
  summarise(min_true = min(true, na.rm=TRUE), max_true = max(true, na.rm=TRUE), .groups="drop") %>%
  left_join(reg_stats, by="set") %>%
  rowwise() %>%
  mutate(
    true_grid   = list(seq(min_true, max_true, length.out = 200)),
    fitted_grid = list(if (is.na(beta)) rep(NA_real_, 200) else pmax(0, beta*true_grid))
  ) %>%
  ungroup() %>%
  unnest(c(true_grid, fitted_grid)) %>%
  transmute(set, lp_true = log1p(true_grid), lp_fitted = log1p(fitted_grid))

# --- Compact annotation: only β and R2_reg(orig) ---
plot_df <- merge(tr_va_te, reg_stats, all.x=T) |> 
  mutate(
    lp_true=log1p(true), lp_pred=log1p(pred),
    pos.x=max(lp_true)*0.68, pos.y=max(lp_pred)*0.98,
    label=sprintf("AUC=%.3f \n β=%.3f \n R2=%.3f ", auc, beta, r2_corr))

# --- Plot: scatter (log1p) + 45° ref + fitted curve ---
ggplot(plot_df, aes(x = lp_true, y = lp_pred)) +
  facet_wrap(. ~ set, nrow = 1) +
  geom_point(color = "darkred", alpha = 0.8, size = 1, shape=1) +
  geom_abline(slope = 1, intercept = 0) +
  geom_text(aes(x = pos.x, y = pos.y, label = label),
            hjust = 1, vjust = 1, size = 3) +
  geom_line(data = fit_lines, aes(x = lp_true, y = lp_fitted),
            inherit.aes = FALSE, color = "darkgreen", linewidth = 1, linetype = "dashed") +
  labs(x = "log1p(true)", y = "log1p(pred)") +
  theme_bw()

ggsave(file.path(getwd(), "results/temp_gate2.png" ), width = 10, height = 5)