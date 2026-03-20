


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
# time consistent party info
# ------------------------------------------------------------------------

# --- 2014
yr <- 2014
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl("partier_2022.xlsx", f, ignore.case = T) ]
cand_raw <- read_xlsx(f, skip = 1) |> clean_names()

# filter valtyp k
cand_raw <- cand_raw |> filter(valtyp=="KF")

# rename
cand_raw <- cand_raw |> 
  rename(
    pty_code=partikod, pty_short=partiforkortning, pty_name=partibeteckning,
  )  |> 
  mutate(
    pty_short=tolower(pty_short), pty_name=tolower(pty_name)
  )

# create a party dict
party_dict <- cand_raw |> select(matches("pty_")) |> drop_na() |> distinct() 
party_dict2014 <- as.tibble(party_dict)

# --- 2018
yr <- 2018
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl("kandid", f, ignore.case = T) & grepl(yr, f) ]
cand_raw <- read_excel(f) %>% clean_names()

# filter valtyp k
cand_raw <- cand_raw |> filter(valtyp=="K")

# rename
cand_raw <- cand_raw |> 
  rename(
    lans_kommun_kod=valomradeskod,
    cand_age=alder_pa_valdagen, cand_sex=kon, 
    pty_code=partikod, pty_short=partiforkortning, pty_name=partibeteckning,
  )  |> 
  mutate(
    pty_short=tolower(pty_short), pty_name=tolower(pty_name)
  )

# party code important
cand_raw <- cand_raw |> 
  mutate(pty_short=ifelse(is.na(pty_short), paste0("x",pty_code), pty_short)) 

# create a party dict
party_dict <- cand_raw |> select(matches("pty_")) |> drop_na() |> distinct() 
party_dict2018 <- as.tibble(party_dict)

# --- 2022
yr <- 2022
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl("partier_2022.xlsx", f, ignore.case = T) ]
cand_raw <- read_xlsx(f, skip = 1) |> clean_names()

# filter valtyp k
cand_raw <- cand_raw |> filter(valtyp=="KF")

# rename
cand_raw <- cand_raw |> 
  rename(
    pty_code=partikod, pty_short=partiforkortning, pty_name=partibeteckning,
  )  |> 
  mutate(
    pty_short=tolower(pty_short), pty_name=tolower(pty_name)
  )

# create a party dict
party_dict <- cand_raw |> select(matches("pty_")) |> drop_na() |> distinct() 
party_dict2022 <- as.tibble(party_dict)

# --- 2026
yr <- 2026
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl("partier_2026.csv", f, ignore.case = T) ]
cand_raw <- fread(f) |> clean_names()

# filter valtyp k
cand_raw <- cand_raw |> filter(valtyp=="KF")

# rename
cand_raw <- cand_raw |> 
  rename(
    pty_code=partikod, pty_short=partiforkortning, pty_name=partibeteckning,
  )  |> 
  mutate(
    pty_short=tolower(pty_short), pty_name=tolower(pty_name)
  )

# create a party dict
party_dict <- cand_raw |> select(matches("pty_")) |> drop_na() |> distinct() 
party_dict2026 <- as.tibble(party_dict)

# ---- for translation to 2026 name and short
party_code_list <- tibble(pty_code=unique(c(
  party_dict2014$pty_code,
  party_dict2018$pty_code,
  party_dict2022$pty_code,
  party_dict2026$pty_code)
) ) 
# try 2022
party_code_list <- left_join(party_code_list, party_dict2022, by = "pty_code")
# try 2018, if missing add name and short
party_code_list <- left_join(party_code_list, party_dict2018, by = "pty_code", suffix = c("_0", "_2")) |> 
  mutate(
    pty_short=ifelse(is.na(pty_short_0) | pty_short_0=="", pty_short_2, pty_short_0),
    pty_name=ifelse(is.na(pty_name_0) | pty_name_0=="", pty_name_2, pty_name_0),
  ) %>%
  select(pty_code, pty_name, pty_short)
# try 2014, if missing add name and short
party_code_list <- left_join(party_code_list, party_dict2014, by = "pty_code", suffix = c("_0", "_2")) |> 
  mutate(
    pty_short=ifelse(is.na(pty_short_0) | pty_short_0=="", pty_short_2, pty_short_0),
    pty_name=ifelse(is.na(pty_name_0) | pty_name_0=="", pty_name_2, pty_name_0),
  ) %>%
  select(pty_code, pty_name, pty_short)

# ------------------------------------------------------------------------
# 2014
# ------------------------------------------------------------------------

# ----  kommun-level vote COUNTS (sheet 'K antal') ----
yr <- 2014
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl(yr, f) ]
f <- f[grepl("kommunval",f, ignore.case = T) & !grepl("~\\$", f)] 
votes_raw <- read_excel(path = f, sheet=1, skip = 2) %>% clean_names() 
sel_col <- grep("ovr_|blank_|og_|vdt", names(votes_raw), value=T)
votes_raw <- votes_raw |> 
  select(!matches(sel_col)) |> 
  rename(
    lans_kod=lan_1, lans_name=lan_3, kommun_kod=kom, kommun_name=kommun,
    tot_valid=rost_giltiga, tot_votes=rostande, tot_elig_voter=rostb,
  )

# sum invalid votes
votes_raw <- votes_raw %>%
  mutate(
    tot_invalid = tot_votes-tot_valid
  ) 

# Identify ID and TOTAL columns (Swedish diacritics were removed by clean_names())
id_cols    <- c(grep("lans_|kommun_", names(votes_raw), value=T, ignore.case = T))
total_cols <- c(grep("tot_", names(votes_raw), value=T, ignore.case = T))
proc_cols <- c(grep("_proc", names(votes_raw), value=T, ignore.case = T))

# Party columns are everything between IDs and totals
party_cols <- setdiff(names(votes_raw), c(id_cols, total_cols, proc_cols))

# Pivot to long: one row per kommun × party with raw vote counts
votes_long <- votes_raw[,c(id_cols,total_cols,party_cols)] %>% 
  pivot_longer(
    cols = all_of(party_cols),
    names_to = "pty_short",
    values_to = "votes"
  ) %>%
  mutate(
    votes = as.numeric(votes),
    votes = ifelse(is.na(votes), 0, votes),
    pty_short = str_to_lower(pty_short),
    pty_short = sub("_tal_2$", "", pty_short),
    pty_short = sub("_tal$", "", pty_short),
    lans_kommun_kod = paste0(lans_kod,kommun_kod) |> as.numeric()
  )

# ---- add party code
temp <- left_join(votes_long, party_dict2014, by = "pty_short")
# try 2018, if missing add code
temp <- left_join(temp, party_dict2018, by = "pty_short", suffix = c("_0", "_2")) |> 
  mutate(
    pty_code=ifelse(is.na(pty_code_0), pty_code_2, pty_code_0),
  ) %>%
   select(-ends_with("_0"), -ends_with("_2"))
# try 2022, if missing add code
temp <- left_join(temp, party_dict2022, by = "pty_short", suffix = c("_0", "_2")) |> 
  mutate(
    pty_code=ifelse(is.na(pty_code_0), pty_code_2, pty_code_0),
  ) %>%
   select(-ends_with("_0"), -ends_with("_2"))
# create  id for missing pty_code
temp <- temp |> mutate(pty_code=ifelse(is.na(pty_code), 99999, pty_code))
votes_long <- temp

# ---- aggregate pty_code 9999
temp <- votes_long |> select(lans_kommun_kod,pty_code, votes) |> 
  group_by(lans_kommun_kod,pty_code) |> 
  dplyr::summarise(votes=sum(votes))

votes_long <- votes_long |> 
  select(all_of(id_cols), all_of(total_cols), pty_code, lans_kommun_kod) |> distinct() |> 
  left_join(temp, by=c("pty_code", "lans_kommun_kod"))

# ---- Read candidate data and build kommun × parti features ----
# for 2014, complete cand info draws from two seperate files. b file is reference

# read b file
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl("kandidaturer_2014_b.xls", f, ignore.case = T) ]
fb <- read_xls(f, col_names=T) %>% clean_names() |> filter(valtyp=="K") |> 
  rename(
    cand_age=alder, cand_sex=kon, pty_code=lista,
    cand_nr=kandidatnr) |> 
  mutate(pty_code=as.numeric(sub("-.*", "", pty_code))) |> 
  select(matches("lans_|kommun_|cand_|pty_")) |> distinct()

# a file
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl("kandidaturer_2014_a.xlsx", f, ignore.case = T) ]
fa <- read_xlsx(f, col_names=F) %>% clean_names()

# rename
fa <- fa |> filter(x1=="K") |> 
  rename(
    lans_kod=x2, kommun_kod = x4, cand_nr=x12
  ) |> 
  mutate(
    lans_kommun_kod=paste0(lans_kod, kommun_kod) |> as.numeric(),
  )

# select vars
fa <- fa |> 
  select(matches("lans_|kommun_|cand_|pty_")) |> distinct()

# join a b file
cand_raw <- fb |> left_join(fa)

# ---- Aggregate candidate features at (kommun × parti) ----
# for continuous var
agg_numeric <- cand_raw |> 
  filter(!is.na(lans_kommun_kod), !is.na(pty_code)) %>%
  group_by(lans_kommun_kod, pty_code) %>%
  dplyr::summarise(
    cand_n      = n(),
    cand_women     = mean(cand_sex=="K", na.rm = TRUE),
    .groups = "drop"
  )

# for categorical var
agg_factor <- cand_raw |> 
  mutate(
    cand_age_group = case_when(
      cand_age >= 18 & cand_age <= 29 ~ "18–29",
      cand_age >= 30 & cand_age <= 49 ~ "30–49",
      cand_age >= 50 & cand_age <= 64 ~ "50–64",
      cand_age >= 65                  ~ "65+",
      TRUE ~ "missing"
    ),
    cand_age_group = factor(
      cand_age_group,
      levels = c("18–29", "30–49", "50–64", "65+", "missing")
    )
  ) %>%
  count(lans_kommun_kod, pty_code, cand_age_group) |> 
  filter(!is.na(lans_kommun_kod), !is.na(pty_code)) %>%
  group_by(lans_kommun_kod, pty_code) %>%
  complete(cand_age_group, fill = list(n = 0)) %>%   
  mutate(prop = n / sum(n)) %>%
  ungroup() |> select(-n) |> 
  pivot_wider(
    id_cols = c(lans_kommun_kod, pty_code),
    names_from = cand_age_group, values_from = prop,
    names_prefix = "cand_age_", values_fill = 0
  )

cand_aggr <- full_join(agg_factor,agg_numeric,by=c("lans_kommun_kod", "pty_code"))
  
# ---- Join kommun votes with candidate features ----
votes_long2 <- votes_long 

votes_long2 <- left_join(votes_long2, cand_aggr, by = c("lans_kommun_kod", "pty_code") )

# handle NAs, if pty_short==other, 0
votes_long2 <- votes_long2 %>%
  mutate(
    has_cand = ifelse(is.na(cand_n), 0L, 1L),
    has_pty_code = ifelse(!is.na(pty_code), 1L, 0L),

    cand_n = replace_na(cand_n, 0),,
    cand_women = replace_na(cand_women, 0)
  ) |> 
  mutate(across(starts_with("cand_age"),  ~ replace_na(., 0)                      
))

# merge party dict 
votes_long2 <- votes_long2 |> left_join(party_code_list)

fwrite(votes_long2, file.path(getwd(), "input_data", paste0(yr, "_clean_cand.csv") ))

# ------------------------------------------------------------------------
# 2018
# ------------------------------------------------------------------------

# ----  kommun-level vote COUNTS (sheet 'K antal') ----
yr <- 2018
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl(yr, f) ]
f <- f[grepl("k_per",f, ignore.case = T) & !grepl("~\\$", f)] 
votes_raw <- read_excel(path = f, sheet="K antal") %>% clean_names() |> 
  select(-c(valdeltagande,ogej,blank,og)) |> 
  rename(
    lans_kod=lanskod, lans_name=lansnamn, kommun_kod=kommunkod, kommun_name=kommunnamn,
    tot_valid=roster_giltiga, tot_votes=rostande, tot_elig_voter=rostberattigade,
  )

# sum invalid votes
votes_raw <- votes_raw %>%
  mutate(
    tot_invalid = tot_votes-tot_valid
  ) 

# Identify ID and TOTAL columns (Swedish diacritics were removed by clean_names())
id_cols    <- c(grep("lans_|kommun_", names(votes_raw), value=T, ignore.case = T))
total_cols <- c(grep("tot_", names(votes_raw), value=T, ignore.case = T))

# Party columns are everything between IDs and totals
party_cols <- setdiff(names(votes_raw), c(id_cols, total_cols))

# Pivot to long: one row per kommun × party with raw vote counts
votes_long <- votes_raw %>%
  pivot_longer(
    cols = all_of(party_cols),
    names_to = "pty_short",
    values_to = "votes"
  ) %>%
  mutate(
    votes = as.numeric(votes),
    votes = ifelse(is.na(votes), 0, votes),
    pty_short = str_to_lower(pty_short),
    pty_short = sub("_tal$", "", pty_short),
    lans_kommun_kod = paste0(lans_kod,kommun_kod) |> as.numeric()
  )

# ---- add party code
temp <- left_join(votes_long, party_dict2018, by = "pty_short")
# try 2022, if missing add code
temp <- left_join(temp, party_dict2022, by = "pty_short", suffix = c("_0", "_2")) |> 
  mutate(
    pty_code=ifelse(is.na(pty_code_0), pty_code_2, pty_code_0),
  ) %>%
   select(-ends_with("_0"), -ends_with("_2"))
# create  id for missing pty_code
temp <- temp |> mutate(pty_code=ifelse(is.na(pty_code), 99999, pty_code))
votes_long <- temp

# ---- aggregate pty_code 9999
temp <- votes_long |> select(lans_kommun_kod,pty_code, votes) |> 
  group_by(lans_kommun_kod,pty_code) |> 
  dplyr::summarise(votes=sum(votes))

votes_long <- votes_long |> 
  select(all_of(id_cols), all_of(total_cols), pty_code, lans_kommun_kod) |> distinct() |> 
  left_join(temp, by=c("pty_code", "lans_kommun_kod"))

# ---- Read candidate data and build kommun × parti features ----
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl("kandid", f, ignore.case = T) & grepl(yr, f) ]
cand_raw <- read_excel(f) %>% clean_names()

# filter valtyp k
cand_raw <- cand_raw |> filter(valtyp=="K")

# rename
cand_raw <- cand_raw |> 
  rename(
    lans_kommun_kod=valomradeskod,
    cand_age=alder_pa_valdagen, cand_sex=kon, 
    pty_code=partikod, cand_nr=kandidatnummer
  )  

# select vars
cand_raw <- cand_raw |> 
  select(matches("lans_|kommun_|cand_|pty_")) |> distinct()

# ---- Aggregate candidate features at (kommun × parti) ----
# for continuous var
agg_numeric <- cand_raw |> 
  filter(!is.na(lans_kommun_kod), !is.na(pty_code)) %>%
  group_by(lans_kommun_kod, pty_code) %>%
  dplyr::summarise(
    cand_n      = n(),
    cand_women     = mean(cand_sex=="K", na.rm = TRUE),
    .groups = "drop"
  )


# for categorical var
agg_factor <- cand_raw |> 
  mutate(
    cand_age_group = case_when(
      cand_age >= 18 & cand_age <= 29 ~ "18–29",
      cand_age >= 30 & cand_age <= 49 ~ "30–49",
      cand_age >= 50 & cand_age <= 64 ~ "50–64",
      cand_age >= 65                  ~ "65+",
      TRUE ~ "missing"
    ),
    cand_age_group = factor(
      cand_age_group,
      levels = c("18–29", "30–49", "50–64", "65+", "missing")
    )
  ) %>%
  count(lans_kommun_kod, pty_code, cand_age_group) |> 
  filter(!is.na(lans_kommun_kod), !is.na(pty_code)) %>%
  group_by(lans_kommun_kod, pty_code) %>%
  complete(cand_age_group, fill = list(n = 0)) %>%   
  mutate(prop = n / sum(n)) %>%
  ungroup() |> select(-n) |> 
  pivot_wider(
    id_cols = c(lans_kommun_kod, pty_code),
    names_from = cand_age_group, values_from = prop,
    names_prefix = "cand_age_", values_fill = 0
  )

cand_aggr <- full_join(agg_factor,agg_numeric,by=c("lans_kommun_kod", "pty_code"))
  
# ---- Join kommun votes with candidate features ----
votes_long2 <- votes_long 

votes_long2 <- left_join(votes_long2, cand_aggr, by = c("lans_kommun_kod", "pty_code") )

# handle NAs, if pty_short==other, 0
votes_long2 <- votes_long2 %>%
  mutate(
    has_cand = ifelse(is.na(cand_n), 0L, 1L),
    has_pty_code = ifelse(!is.na(pty_code), 1L, 0L),

    cand_n = replace_na(cand_n, 0),,
    cand_women = replace_na(cand_women, 0)
  ) |> 
  mutate(across(starts_with("cand_age"),  ~ replace_na(., 0)                      
))

# merge party dict 
votes_long2 <- votes_long2 |> left_join(party_code_list)

fwrite(votes_long2, file.path(getwd(), "input_data", paste0(yr, "_clean_cand.csv") ))


# ------------------------------------------------------------------------
# 2022
# ------------------------------------------------------------------------

# ----  kommun-level vote COUNTS (sheet 'K antal') ----
yr <- 2022
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl(yr, f) ]
f <- f[grepl("2022_kommunaval_per_kommun",f, ignore.case = T) & !grepl("~\\$", f)] 
votes_raw <- read_excel(path = f, sheet=2) %>% clean_names() 

# drop unwanted cols
drop_dict <- grep(
  "ej anmält deltagande|blanka röster|övriga ogiltiga",
  unique(votes_raw$parti),value=T, ignore.case=T)
votes_raw <- votes_raw |> filter(!parti %in% drop_dict)

# create id cols
votes_raw <- votes_raw |> 
  mutate(
    lans_kod=sub("^KF-([0-9]+)-.*$", "\\1", distrikt) |> as.numeric(), 
    lans_name=lan, 
    kommun_kod=sub("^KF-[0-9]+-([0-9]+)-.*$", "\\1", distrikt) |> as.numeric(), 
    kommun_name=kommun,
  ) 

# aggregate
tot_elig_voter <- votes_raw |> 
  select(lans_kod,lans_name,kommun_kod,kommun_name, rostberattigade) |> 
  distinct() |> 
  group_by(lans_kod,lans_name,kommun_kod,kommun_name) |> 
  dplyr::summarise(tot_elig_voter=sum(rostberattigade))

votes_raw <- votes_raw |> 
  group_by(lans_kod,lans_name,kommun_kod,kommun_name, parti) |> 
  dplyr::summarise(votes=sum(roster))

# to wide first
votes_raw <- votes_raw |> spread(key=parti, value=votes) |> 
  rename(tot_valid='Summa giltiga röster', tot_votes=Valdeltagande) |> 
  mutate(tot_invalid=tot_votes-tot_valid)

# Identify ID and TOTAL columns (Swedish diacritics were removed by clean_names())
id_cols    <- c(grep("lans_|kommun_", names(votes_raw), value=T, ignore.case = T))
total_cols <- c(grep("tot_", names(votes_raw), value=T, ignore.case = T))

# Party columns are everything between IDs and totals
party_cols <- setdiff(names(votes_raw), c(id_cols, total_cols))

# Pivot to long: one row per kommun × party with raw vote counts
votes_long <- votes_raw %>%
  pivot_longer(
    cols = all_of(party_cols),
    names_to = "pty_name",
    values_to = "votes"
  ) %>%
  mutate(
    votes = as.numeric(votes),
    votes = ifelse(is.na(votes), 0, votes),
    pty_name = str_to_lower(pty_name),
    pty_name = sub("_tal$", "", pty_name),
    lans_kommun_kod = paste0(lans_kod,kommun_kod) |> as.numeric()
  )

# add back eligible voters
votes_long <- votes_long |> left_join(tot_elig_voter)
total_cols <- c(grep("tot_", names(votes_long), value=T, ignore.case = T))

# ---- add party code
temp <- left_join(votes_long, party_dict2022, by = "pty_name")
# try 2022, if missing add code
temp <- left_join(temp, party_dict2026, by = "pty_name", suffix = c("_0", "_2")) |> 
  mutate(
    pty_code=ifelse(is.na(pty_code_0), pty_code_2, pty_code_0),
  ) %>%
   select(-ends_with("_0"), -ends_with("_2"))
# create  id for missing pty_code
temp <- temp |> mutate(pty_code=ifelse(is.na(pty_code), 99999, pty_code))
votes_long <- temp

# ---- aggregate pty_code 9999
temp <- votes_long |> select(lans_kommun_kod,pty_code, votes) |> 
  group_by(lans_kommun_kod,pty_code) |> 
  dplyr::summarise(votes=sum(votes))

votes_long <- votes_long |> 
  select(all_of(id_cols), all_of(total_cols), pty_code, lans_kommun_kod) |> distinct() |> 
  left_join(temp, by=c("pty_code", "lans_kommun_kod"))


# ---- Read candidate data and build kommun × parti features ----
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl("kandid", f, ignore.case = T) & grepl(yr, f) ]
cand_raw <- fread(f) |> clean_names()

# filter valtyp k
cand_raw <- cand_raw |> filter(valtyp=="KF")

# rename
cand_raw <- cand_raw |> 
  rename(
    lans_kommun_kod=valomradeskod,
    cand_age=alder_pa_valdagen, cand_sex=kon, 
    pty_code=partikod, cand_nr=kandidatnummer
  )  

# select vars
cand_raw <- cand_raw |> 
  select(matches("lans_|kommun_|cand_|pty_")) |> distinct()

# ---- Aggregate candidate features at (kommun × parti) ----
# for continuous var
agg_numeric <- cand_raw |> 
  filter(!is.na(lans_kommun_kod), !is.na(pty_code)) %>%
  group_by(lans_kommun_kod, pty_code) %>%
  dplyr::summarise(
    cand_n      = n(),
    cand_women     = mean(cand_sex=="K", na.rm = TRUE),
    .groups = "drop"
  )

# for categorical var
agg_factor <- cand_raw |> 
  mutate(
    cand_age_group = case_when(
      cand_age >= 18 & cand_age <= 29 ~ "18–29",
      cand_age >= 30 & cand_age <= 49 ~ "30–49",
      cand_age >= 50 & cand_age <= 64 ~ "50–64",
      cand_age >= 65                  ~ "65+",
      TRUE ~ "missing"
    ),
    cand_age_group = factor(
      cand_age_group,
      levels = c("18–29", "30–49", "50–64", "65+", "missing")
    )
  ) %>%
  count(lans_kommun_kod, pty_code, cand_age_group) |> 
  filter(!is.na(lans_kommun_kod), !is.na(pty_code)) %>%
  group_by(lans_kommun_kod, pty_code) %>%
  complete(cand_age_group, fill = list(n = 0)) %>%   
  mutate(prop = n / sum(n)) %>%
  ungroup() |> select(-n) |> 
  pivot_wider(
    id_cols = c(lans_kommun_kod, pty_code),
    names_from = cand_age_group, values_from = prop,
    names_prefix = "cand_age_", values_fill = 0
  )

cand_aggr <- full_join(agg_factor,agg_numeric,by=c("lans_kommun_kod", "pty_code"))
  
# ---- Join kommun votes with candidate features ----
votes_long2 <- votes_long 

votes_long2 <- left_join(votes_long2, cand_aggr, by = c("lans_kommun_kod", "pty_code") )

# handle NAs, if pty_short==other, 0
votes_long2 <- votes_long2 %>%
  mutate(
    has_cand = ifelse(is.na(cand_n), 0L, 1L),
    has_pty_code = ifelse(!is.na(pty_code), 1L, 0L),

    cand_n = replace_na(cand_n, 0),,
    cand_women = replace_na(cand_women, 0)
  ) |> 
  mutate(across(starts_with("cand_age"),  ~ replace_na(., 0)                      
))

# merge party dict 
votes_long2 <- votes_long2 |> left_join(party_code_list)

fwrite(votes_long2, file.path(getwd(), "input_data", paste0(yr, "_clean_cand.csv") ))



# ------------------------------------------------------------------------
# 2026 predictor data for forecast, offset, candidate composition, lag votes
# ------------------------------------------------------------------------

# ----  kommun-level vote COUNTS (sheet 'K antal') ----
yr <- 2026
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl(yr, f) ]
f <- f[grepl("partier",f, ignore.case = T) & !grepl("~\\$", f)] 
votes_raw <- fread(f) %>% clean_names() |> as_tibble()

# filter valtyp
votes_raw <- filter(votes_raw, valtyp=="KF")

# create id cols
votes_raw <- votes_raw |> 
  rename(
    lans_kod=lanskod, 
    lans_name=lansnamn, 
    kommun_name=valomradesnamn,
    lans_kommun_kod=valomradeskod,
    pty_code=partikod
  ) |> 
  mutate(kommun_kod=as.numeric(gsub("\\d{2}$", "", lans_kommun_kod)))

# Identify ID and TOTAL columns (Swedish diacritics were removed by clean_names())
id_cols    <- c(grep("lans_|kommun_", names(votes_raw), value=T, ignore.case = T))
total_cols <- c(grep("tot_", names(votes_raw), value=T, ignore.case = T))
party_cols <- c(grep("pty_", names(votes_raw), value=T, ignore.case = T))

votes_raw <- votes_raw |> select(all_of(id_cols), all_of(total_cols), all_of(party_cols)) |> distinct()

# create tot vars (missing in forecasting)
votes_raw <- votes_raw |> mutate(
  tot_valid=NA, tot_votes=NA, tot_invalid=NA, votes=NA
)

# add back eligible voters
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl(yr, f) ]
f <- f[grepl("tot_elig",f, ignore.case = T) & !grepl("~\\$", f)] 
temp <- read_xlsx(f, sheet = "Antal röstberättigade") %>% clean_names() |> as_tibble() |> 
  mutate(lans_kommun_kod=kommunkod |> as.numeric()) |> 
  rename(
    tot_elig_voter=rostberattigade_val_till_kommunfullmaktige 
  ) |> 
  select(matches("lans_|tot_")) |> 
  group_by(lans_kommun_kod) |> dplyr::summarise(tot_elig_voter=sum(tot_elig_voter))
votes_long <- left_join(votes_raw,temp, by="lans_kommun_kod")

total_cols <- c(grep("tot_", names(votes_long), value=T, ignore.case = T))

# ---- Read candidate data and build kommun × parti features ----
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl("kandid", f, ignore.case = T) & grepl(yr, f) ]
cand_raw <- fread(f) |> clean_names()

# filter valtyp k
cand_raw <- cand_raw |> filter(valtyp=="KF")

# rename
cand_raw <- cand_raw |> 
  rename(
    lans_kommun_kod=valomradeskod,
    cand_age=alder_pa_valdagen, cand_sex=kon, 
    pty_code=partikod, cand_nr=kandidatnummer
  )  

# select vars
cand_raw <- cand_raw |> 
  select(matches("lans_|kommun_|cand_|pty_")) |> distinct()

# ---- Aggregate candidate features at (kommun × parti) ----
# for continuous var
agg_numeric <- cand_raw |> 
  filter(!is.na(lans_kommun_kod), !is.na(pty_code)) %>%
  group_by(lans_kommun_kod, pty_code) %>%
  dplyr::summarise(
    cand_n      = n(),
    cand_women     = mean(cand_sex=="K", na.rm = TRUE),
    .groups = "drop"
  )

# for categorical var
agg_factor <- cand_raw |> 
  mutate(
    cand_age_group = case_when(
      cand_age >= 18 & cand_age <= 29 ~ "18–29",
      cand_age >= 30 & cand_age <= 49 ~ "30–49",
      cand_age >= 50 & cand_age <= 64 ~ "50–64",
      cand_age >= 65                  ~ "65+",
      TRUE ~ "missing"
    ),
    cand_age_group = factor(
      cand_age_group,
      levels = c("18–29", "30–49", "50–64", "65+", "missing")
    )
  ) %>%
  count(lans_kommun_kod, pty_code, cand_age_group) |> 
  filter(!is.na(lans_kommun_kod), !is.na(pty_code)) %>%
  group_by(lans_kommun_kod, pty_code) %>%
  complete(cand_age_group, fill = list(n = 0)) %>%   
  mutate(prop = n / sum(n)) %>%
  ungroup() |> select(-n) |> 
  pivot_wider(
    id_cols = c(lans_kommun_kod, pty_code),
    names_from = cand_age_group, values_from = prop,
    names_prefix = "cand_age_", values_fill = 0
  )

cand_aggr <- full_join(agg_factor,agg_numeric,by=c("lans_kommun_kod", "pty_code"))
  
# ---- Join kommun votes with candidate features ----
votes_long2 <- votes_long 

votes_long2 <- left_join(votes_long2, cand_aggr, by = c("lans_kommun_kod", "pty_code") )

# handle NAs, if pty_short==other, 0
votes_long2 <- votes_long2 %>%
  mutate(
    has_cand = ifelse(is.na(cand_n), 0L, 1L),
    has_pty_code = ifelse(!is.na(pty_code), 1L, 0L),

    cand_n = replace_na(cand_n, 0),,
    cand_women = replace_na(cand_women, 0)
  ) |> 
  mutate(across(starts_with("cand_age"),  ~ replace_na(., 0)                      
))

# merge party dict 
votes_long2 <- votes_long2 |> left_join(party_code_list)

fwrite(votes_long2, file.path(getwd(), "input_data", paste0(yr, "_clean_cand.csv") ))


