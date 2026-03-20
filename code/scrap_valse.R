
# ------------------------------------------------------------------------
# config.
# ------------------------------------------------------------------------

# If running in RStudio, set working dir to script location (optional)
if (requireNamespace("rstudioapi", quietly = TRUE) &&
  rstudioapi::isAvailable()) {
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  }

library(rvest)
library(httr)
library(readr)
library(stringr)

dest_file <- file.path(getwd(),"input_data")

# ------------------------------------------------------------------------
# web 
# ------------------------------------------------------------------------
# Page URL
page_url <- "https://www.val.se/valresultat-och-statistik/statistik-och-data/radata-val-2026"

# Read webpage
page <- read_html(page_url)

# Find all links
links <- page %>%
  html_elements("a") %>%
  html_attr("href")

# ------------------------------------------------------------------------
# download candidate file
# ------------------------------------------------------------------------
# Filter for the correct CSV file
target_link <- links[str_detect(links, "kandidaturer")]

# Handle relative URLs
base_url <- "https://www.val.se"
download_url <- ifelse(str_detect(target_link, "^http"),
                       target_link,
                       paste0(base_url, target_link))

# Download file
download.file(download_url, destfile = file.path(dest_file, "kandidaturer_2026.csv"), mode = "wb")


# ------------------------------------------------------------------------
# download parti file
# ------------------------------------------------------------------------
# Filter for the correct CSV file
target_link <- links[str_detect(links, "deltagande-partier")]

# Handle relative URLs
base_url <- "https://www.val.se"
download_url <- ifelse(str_detect(target_link, "^http"),
                       target_link,
                       paste0(base_url, target_link))

# Download file
download.file(download_url, destfile = file.path(dest_file, "partier_2026.csv"), mode = "wb")


# ------------------------------------------------------------------------
# download elig voter file
# ------------------------------------------------------------------------
# Filter for the correct CSV file
target_link <- links[str_detect(links, "antal-rostberattigade-per")]

# Extract timestamp (the long number in the URL)
df <- data.frame(link = target_link) %>%
  mutate(
    timestamp = str_extract(link, "\\d{13}"),
    timestamp = as.numeric(timestamp)
  )

# Pick latest
latest_link <- df %>%
  filter(!is.na(timestamp)) %>%
  slice_max(timestamp, n = 1) %>%
  pull(link)

# Build full URL
download_url <- paste0("https://www.val.se", latest_link)

# Download file
download.file(download_url, destfile = file.path(dest_file, "tot_elig_voter_2026.csv"), mode = "wb")

