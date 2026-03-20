


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

# ----  kommun-level eligible voter ----
yr <- 2018
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl(yr, f) ]
f <- f[grepl("voter_age_sex",f, ignore.case = T) & !grepl("~\\$", f)] 
df_raw <- read_excel(path = f, sheet=1) %>% clean_names() |> 
  select(!matches("valdistrikt|valkret")) |> 
  rename(
    lans_kod=lan_id, lans_name=lan_namn, lans_kommun_kod=kommun_id, kommun_name=kommun_namn
  )

# Identify ID and TOTAL columns (Swedish diacritics were removed by clean_names())
geo    <- c(grep("lans_|kommun_", names(df_raw), value=T, ignore.case = T))

# Pivot to long: one row per kommun × party with raw vote counts
df_long <- df_raw %>%
  pivot_longer(
    cols = setdiff(names(df_raw), id_cols),
    names_to = "key",
    values_to = "voter"
  ) 

# ---- parse key ----
setDT(df_long)

pat <- "^(man|kvinnor)_((?:ej_)?svenska_medborgare)_(18_29|30_49|50_64|65)(?:_((?:ej_)?forstagangsvaljare))?$"
m <- str_match(df_long$key, pat)

df_long[, `:=`(
  vot_sex       = m[,2],
  vot_citizen   = m[,3],
  vot_age       = m[,4],
  vot_first_time   = m[,5]
)]

df_long[, `:=`(
  vot_citizen = fcase(
    vot_citizen == "svenska_medborgare",    "svensk_medborgare",
    vot_citizen == "ej_svenska_medborgare", "ej_svensk_medborgare",
    default = NA_character_
  ),
  vot_age = fcase(
    vot_age == "18_29", "18–29",
    vot_age == "30_49", "30–49",
    vot_age == "50_64", "50–64",
    vot_age == "65",    "65+",
    default = NA_character_
  ),
  vot_first_time = fcase(
    vot_first_time == "forstagangsvaljare",        TRUE,
    vot_first_time == "ej_forstagangsvaljare",     FALSE,
    is.na(vot_first_time),                         FALSE,
    default = NA
  )
)]

df_long[, key := NULL]  # drop intermediate

# ---- aggregate ----

# 1) Totals per kommun
tot <- df_long[, .(voters_total = sum(voter, na.rm = TRUE)), by = geo]

# 2) Numerators
# 2a) Women
num_women <- df_long[vot_sex == "kvinnor",
  .(voters_women = sum(voter, na.rm = TRUE)),
  by = geo
]

# 2b) Swedish citizens
num_swe <- df_long[vot_citizen == "svensk_medborgare",
  .(voters_swe_citizen = sum(voter, na.rm = TRUE)),
  by = geo
]

# 2c) Age groups → wide (one column per age group)
age_counts_long <- df_long[
  , .(voters_age = sum(voter, na.rm = TRUE)),
  by = c(geo, "vot_age")
]

# Safe column names for ages (avoid '+' in column names)
age_map <- c("18–29" = "age_18_29", "30–49" = "age_30_49",
             "50–64" = "age_50_64", "65+" = "age_65plus")
age_counts_long[, age_col := age_map[vot_age]]

age_counts_wide <- dcast(
  age_counts_long,
  lans_kod + lans_name + lans_kommun_kod + kommun_name ~ age_col,
  value.var = "voters_age",
  fun.aggregate = sum,
  fill = 0
)

# 2d) First-time voters (across all rows where vot_first_time==TRUE)
num_first <- df_long[vot_first_time %in% TRUE,
  .(voters_first_time = sum(voter, na.rm = TRUE)),
  by = geo
]

# 3) Combine numerators with totals
demo_counts <- Reduce(function(x, y) merge(x, y, by = geo, all = TRUE),
                      list(tot, num_women, num_swe, age_counts_wide, num_first))

# Replace missing numerators with 0
for (col in setdiff(names(demo_counts), geo)) {
  set(demo_counts, i = which(is.na(demo_counts[[col]])), j = col, value = 0)
}

# 4) Turn counts into proportions
demo_props <- copy(demo_counts)
demo_props[
  , `:=`(
      # overall shares
      vot_women          = voters_women        / pmax(voters_total, 1),
      vot_swe_citizen    = voters_swe_citizen  / pmax(voters_total, 1),
      vot_age_18_29      = age_18_29           / pmax(voters_total, 1),
      vot_age_30_49      = age_30_49           / pmax(voters_total, 1),
      vot_age_50_64      = age_50_64           / pmax(voters_total, 1),
      vot_age_65plus     = age_65plus          / pmax(voters_total, 1),
      vot_first_time = voters_first_time   / pmax(voters_total, 1)
    )
]

# 5) Keep just the proportions (and, if you want, the raw totals)
vot_cols <- grep("^vot_", names(demo_props), value = TRUE)
demo_props_keep <- demo_props[, c(geo, vot_cols), with = FALSE]

fwrite(demo_props_keep, file.path(getwd(), "input_data", paste0(yr, "_clean_voter.csv") ))



# ------------------------------------------------------------------------
# 2022
# ------------------------------------------------------------------------

# ----  kommun-level eligible voter ----
yr <- 2022
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl(yr, f) ]
f <- f[grepl("voter_age_sex",f, ignore.case = T) & !grepl("~\\$", f)] 
df_raw <- read_excel(path = f, sheet=1, 
  skip  = 2,           # <--- key change: skip the title row
  col_types = "text"   # keep codes like 0114, 01 as strings
) %>% clean_names() |> 
  select(!matches("valdistrikt|valkret")) |> 
  rename(
    lans_kod=lanskod, lans_name=lan, lans_kommun_kod=kommunkod, kommun_name=kommun
  ) |> 
  mutate(lans_kod=as.numeric(lans_kod), lans_kommun_kod=as.numeric(lans_kommun_kod)) |> 
  filter(!is.na(lans_kod) & !is.na(lans_kommun_kod))

# Identify ID and TOTAL columns (Swedish diacritics were removed by clean_names())
geo    <- c(grep("lans_|kommun_", names(df_raw), value=T, ignore.case = T))

# Pivot to long: one row per kommun × party with raw vote counts
df_long <- df_raw %>%
  pivot_longer(
    cols = setdiff(names(df_raw), geo),
    names_to = "key",
    values_to = "voter"
  ) 

# ---- parse key ----
setDT(df_long)

pat <- "^(man|kvinnor)_((?:ej_)?svenska_medborgare)_(18_29|30_49|50_64|65)(?:_((?:ej_)?forstagangsvaljare))?$"
m <- str_match(df_long$key, pat)

df_long[, `:=`(
  vot_sex       = m[,2],
  vot_citizen   = m[,3],
  vot_age       = m[,4],
  vot_first_time   = m[,5]
)]

df_long[, `:=`(
  vot_citizen = fcase(
    vot_citizen == "svenska_medborgare",    "svensk_medborgare",
    vot_citizen == "ej_svenska_medborgare", "ej_svensk_medborgare",
    default = NA_character_
  ),
  vot_age = fcase(
    vot_age == "18_29", "18–29",
    vot_age == "30_49", "30–49",
    vot_age == "50_64", "50–64",
    vot_age == "65",    "65+",
    default = NA_character_
  ),
  vot_first_time = fcase(
    vot_first_time == "forstagangsvaljare",        TRUE,
    vot_first_time == "ej_forstagangsvaljare",     FALSE,
    is.na(vot_first_time),                         FALSE,
    default = NA
  )
)]

df_long[, key := NULL]  # drop intermediate

# ---- aggregate ----
df_long[, voter := as.numeric(voter)]

# 1) Totals per kommun
tot <- df_long[, .(voters_total = sum(voter, na.rm = TRUE)), by = geo]

# 2) Numerators
# 2a) Women
num_women <- df_long[vot_sex == "kvinnor",
  .(voters_women = sum(voter, na.rm = TRUE)),
  by = geo
]

# 2b) Swedish citizens
num_swe <- df_long[vot_citizen == "svensk_medborgare",
  .(voters_swe_citizen = sum(voter, na.rm = TRUE)),
  by = geo
]

# 2c) Age groups → wide (one column per age group)
age_counts_long <- df_long[
  , .(voters_age = sum(voter, na.rm = TRUE)),
  by = c(geo, "vot_age")
]

# Safe column names for ages (avoid '+' in column names)
age_map <- c("18–29" = "age_18_29", "30–49" = "age_30_49",
             "50–64" = "age_50_64", "65+" = "age_65plus")
age_counts_long[, age_col := age_map[vot_age]]

age_counts_wide <- dcast(
  age_counts_long,
  lans_kod + lans_name + lans_kommun_kod + kommun_name ~ age_col,
  value.var = "voters_age",
  fun.aggregate = sum,
  fill = 0
)

# 2d) First-time voters (across all rows where vot_first_time==TRUE)
num_first <- df_long[vot_first_time %in% TRUE,
  .(voters_first_time = sum(voter, na.rm = TRUE)),
  by = geo
]

# 3) Combine numerators with totals
demo_counts <- Reduce(function(x, y) merge(x, y, by = geo, all = TRUE),
                      list(tot, num_women, num_swe, age_counts_wide, num_first))

# Replace missing numerators with 0
for (col in setdiff(names(demo_counts), geo)) {
  set(demo_counts, i = which(is.na(demo_counts[[col]])), j = col, value = 0)
}

# 4) Turn counts into proportions
demo_props <- copy(demo_counts)
demo_props[
  , `:=`(
      # overall shares
      vot_women          = voters_women        / pmax(voters_total, 1),
      vot_swe_citizen    = voters_swe_citizen  / pmax(voters_total, 1),
      vot_age_18_29      = age_18_29           / pmax(voters_total, 1),
      vot_age_30_49      = age_30_49           / pmax(voters_total, 1),
      vot_age_50_64      = age_50_64           / pmax(voters_total, 1),
      vot_age_65plus     = age_65plus          / pmax(voters_total, 1),
      vot_first_time = voters_first_time   / pmax(voters_total, 1)
    )
]

# 5) Keep just the proportions (and, if you want, the raw totals)
vot_cols <- grep("^vot_", names(demo_props), value = TRUE)
demo_props_keep <- demo_props[, c(geo, vot_cols), with = FALSE]

fwrite(demo_props_keep, file.path(getwd(), "input_data", paste0(yr, "_clean_voter.csv") ))

