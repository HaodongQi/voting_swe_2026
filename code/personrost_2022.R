
# ------------------------------------
# config.
# ------------------------------------

# If running in RStudio, set working dir to script location (optional)
if (requireNamespace("rstudioapi", quietly = TRUE) &&
  rstudioapi::isAvailable()) {
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  }

pacman::p_load(lavaan, semPlot, data.table, tidyverse, ggcorrplot,
  ComplexHeatmap, circlize, readxl)

# ------------------------------------
# data
# ------------------------------------
f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl("kandid", f) & grepl("2022", f) ]
candinfo <- fread(f)
candinfo <- candinfo |> as_tibble() |> filter(VALTYP=="KF") |> 
  dplyr::rename(
    cand_nr=KANDIDATNUMMER, cand_age=ÅLDER_PÅ_VALDAGEN, cand_sex=KÖN
  ) |> 
  dplyr::select(matches("cand_")) |> distinct() |> drop_na()

f <- list.files(file.path(getwd(),"input_data"), full.names = T)
f <- f[grepl("Personroster", f) & grepl("2022", f) ]
votes <- read_excel(f, sheet = "Rådata") 
votes <- votes |> filter(Valtyp=="KF") |> 
  dplyr::rename(
    cand_nr=Kandidatnr, cand_votes='Antal personröster', cand_name=Namn, 
    cand_llkk='Län/Kommun', cand_lan=Län, cand_kommun=Kommun, 
    cand_party=Parti, cand_partycode=Partikod
  ) |> 
  dplyr::select(matches("cand_")) 

temp <- votes |> select(!matches("votes")) |> distinct()
votes <- votes |> group_by(cand_nr) |> 
  dplyr::summarise(cand_votes=sum(cand_votes, na.rm = T)) |> ungroup() |> 
  left_join(temp)

votes <- votes |> left_join(candinfo, by=c("cand_nr"))

fwrite(votes, file.path(getwd(), "votes_2022.csv"))


####################################################################################
v1 <- c(0,0,1,0)
v2 <- c(.1, .1, .5, .3)
# v1 is one-hot, v2 sums to 1
party_index <- which(v1 == 1)
# Bhattacharyya coefficient
sqrt(v2[party_index])

