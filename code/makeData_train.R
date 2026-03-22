

# ------------------------------------------------------------------------
# config.
# ------------------------------------------------------------------------

# If running in RStudio, set working dir to script location (optional)
if (requireNamespace("rstudioapi", quietly = TRUE) &&
  rstudioapi::isAvailable()) {
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  }

pacman::p_load(readxl,tidyverse,janitor,stringi,stringr,lubridate, data.table,
  stringdist,mpath,rsample,yardstick,fastDummies)

# ------------------------------------------------------------------------
# stack all years
# ------------------------------------------------------------------------
yr <- c(2014,2018,2022, 2026)

out <- data.table()

for(i in yr){

  # ----  votes and party composition ----
  f <- list.files(file.path(getwd(),"input_data"), full.names = T)
  f <- f[grepl(i, f) ]
  f <- f[grepl("clean",f, ignore.case = T) & !grepl("~\\$", f)] 

  df <- fread(f[grepl("cand",f, ignore.case = T) & !grepl("~\\$", f)] )

  id_cols <-  grep("lans_kommun_", names(df), value = TRUE)

  # ----  voter composition ----
  temp <- fread(f[grepl("voter",f, ignore.case = T) & !grepl("~\\$", f)] )
  keep_cols <- c(
    id_cols, 
    grep("vot_", names(temp), value = TRUE))
  temp <- temp[, ..keep_cols]

  df <- merge(df, temp, by=id_cols, all.x = T)

  # NA to 0
  vot_cols <- grep("^vot_", names(df), value = TRUE)
  df[ , 
    (vot_cols) := lapply(.SD, function(x) {x=fifelse(is.na(x),0,x)}), 
    .SDcols = vot_cols]

  # has ?
  df[ , has_missing_age := fifelse(cand_age_missing==0,0,1)]

  # ----  party voter similarity ----

  # Candidate & voter age variables in matching order
  df[, vot_age_missing := 0]

  cand_age_vars <- c(
    "cand_age_18–29", "cand_age_30–49", "cand_age_50–64", "cand_age_65+", "cand_age_missing")
  vot_age_vars  <- c(
    "vot_age_18_29",  "vot_age_30_49",  "vot_age_50_64",  "vot_age_65plus", "vot_age_missing")

  eps <- 1e-8

  df[, sim_age := {
    C <- as.matrix(.SD[, ..cand_age_vars])
    V <- as.matrix(.SD[, ..vot_age_vars])

    C <- sweep(C, 1, rowSums(C) + eps * ncol(C), "/")
    V <- sweep(V, 1, rowSums(V) + eps * ncol(V), "/")

    BC <- rowSums(sqrt(C * V))      # in [0,1]
    pmax(0, pmin(1, BC))
  }]

  # women sim
  df[, sim_women := 1 - abs(cand_women - vot_women)]
  df[vot_women == 0, sim_women := 0]  # if you treat 0 as missing; remove if 0 can be real

  # add year
  df[, time:=i]

  out <- rbind(out, df)

}


# ------------------------------------------------------------------------
# lag vars 
# ------------------------------------------------------------------------

setDT(out)

# 0) (Optional) sanity check for duplicates:
dups <- out[, .N, by = .(lans_kommun_kod, pty_code, time)][N > 1]
if (nrow(dups)) {
  message("WARNING: duplicates detected for (kommun, party, time).")
  print(head(dups))
  # If needed, resolve duplicates here (e.g., sum votes). Example code:
  # out <- out[, .(
  #   tot_valid = first(tot_valid),   # or sum/mean depending on semantics
  #   tot_votes = first(tot_votes),
  #   tot_elig_voter = first(tot_elig_voter),
  #   tot_invalid = first(tot_invalid),
  #   votes = sum(votes, na.rm = TRUE),  # if duplicates are true splits
  #   # ... aggregate other columns similarly ...
  # ), by = .(lans_kommun_kod, pty_code, time)]
}

# 1) Prepare previous-year (time-4) table to join
prev <- out[, .(
  lans_kommun_kod,
  pty_code,
  time = time + 4L,    # the 'next' election year
  lag_votes = votes,
  # (Optional) if you want lagged denominator for diagnostics:
  lag_tot_elig_voter = tot_elig_voter
)]

setkey(prev, lans_kommun_kod, pty_code, time)
setkey(out,  lans_kommun_kod, pty_code, time)

# 2) Join prev→current on (kommun, party, time == time_prev+4)
out <- prev[out]  # left join 'out' onto 'prev' by keys (adds lag_votes to 'out')

# 3) Flags and normalized lag (note: divide by CURRENT-year tot_elig_voter)
out[, has_lag_vote := as.integer(!is.na(lag_votes))]
out[, lag_rate      := ifelse(!is.na(lag_votes) & lag_tot_elig_voter > 0,
                              lag_votes / lag_tot_elig_voter, NA_real_)]
out[, lag_log_votes := ifelse(!is.na(lag_votes), log1p(lag_votes), NA_real_)]

# 4) Sanity checks (spot check a kommun/party across time)
out[lans_kommun_kod == 114 & pty_code == 1, .(time, votes, lag_votes, has_lag_vote, lag_rate)]


fwrite(out, file.path(getwd(), "input_data", paste0("all_year_clean.csv") ))

