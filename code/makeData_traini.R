

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
# 2018
# ------------------------------------------------------------------------

# ----  votes and party composition ----
yr <- 2018
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl(yr, f) ]
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

# has party code?
df[ , has_pty_info := fifelse(is.na(pty_code),0,1)]
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


# ---- add lag votes ----
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl(2014, f) ]
lag_vote <- fread(f[grepl("clean_cand",f, ignore.case = T) & !grepl("~\\$", f)] )
lag_vote <- lag_vote[, votes:=votes/tot_elig_voter]

selcol <- c("lans_kommun_kod", "pty_short", "votes")
temp <- copy(lag_vote)[, ..selcol]
setnames(temp, "votes", "try1")
df <- merge(df, temp, by= c("lans_kommun_kod", "pty_short"), all.x = T)

selcol <- c("lans_kommun_kod", "pty_code", "votes")
temp <- copy(lag_vote)[, ..selcol]
setnames(temp, "votes", "try2")
temp <- subset(temp, !is.na(pty_code))
df <- merge(df, temp, by= c("lans_kommun_kod", "pty_code"), all.x = T)

selcol <- c("lans_kommun_kod", "pty_name", "votes")
temp <- copy(lag_vote)[, ..selcol]
setnames(temp, "votes", "try3")
temp <- subset(temp, pty_name!="")
df <- merge(df, temp, by= c("lans_kommun_kod", "pty_name"), all.x = T)

df[, lag_votes := do.call(pmax, c(.SD, na.rm=TRUE)), .SDcols = patterns("^try[1-3]$")]
df[, lag_votes:=fifelse(is.na(lag_votes), 0, lag_votes)]

df[, (grep("^try", names(df), value = TRUE)) := NULL]

fwrite(df, file.path(getwd(), "input_data", paste0(yr, "_clean_all.csv") ))

# ------------------------------------------------------------------------
# 2022
# ------------------------------------------------------------------------

# ----  votes and party composition ----
yr <- 2022
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl(yr, f) ]
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

# has party code?
df[ , has_pty_info := fifelse(is.na(pty_code),0,1)]
df[ , has_missing_age := fifelse(cand_age_missing==0,0,1)]

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


# ---- add lag votes ----
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl(2018, f) ]
lag_vote <- fread(f[grepl("clean_cand",f, ignore.case = T) & !grepl("~\\$", f)] )
lag_vote <- lag_vote[, votes:=votes/tot_elig_voter]

selcol <- c("lans_kommun_kod", "pty_short", "votes")
temp <- copy(lag_vote)[, ..selcol]
setnames(temp, "votes", "try1")
df <- merge(df, temp, by= c("lans_kommun_kod", "pty_short"), all.x = T)

selcol <- c("lans_kommun_kod", "pty_code", "votes")
temp <- copy(lag_vote)[, ..selcol]
setnames(temp, "votes", "try2")
temp <- subset(temp, !is.na(pty_code))
df <- merge(df, temp, by= c("lans_kommun_kod", "pty_code"), all.x = T)

selcol <- c("lans_kommun_kod", "pty_name", "votes")
temp <- copy(lag_vote)[, ..selcol]
setnames(temp, "votes", "try3")
temp <- subset(temp, pty_name!="")
df <- merge(df, temp, by= c("lans_kommun_kod", "pty_name"), all.x = T)

df[, lag_votes := do.call(pmax, c(.SD, na.rm=TRUE)), .SDcols = patterns("^try[1-3]$")]
df[, lag_votes:=fifelse(is.na(lag_votes), 0, lag_votes)]

df[, (grep("^try", names(df), value = TRUE)) := NULL]

fwrite(df, file.path(getwd(), "input_data", paste0(yr, "_clean_all.csv") ))