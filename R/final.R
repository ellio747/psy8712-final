# Script settings and resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)            # data science tools
library(httr)                 # http calls
library(jsonlite)             # use of JSON functionality

# Data Import and Cleaning
final_tbl <- read_csv("../data/glassdoor_reviews.csv") # import dataset

get_embedding <- function(text_strings) { #returns embedding vector for any string
  response <- POST(
    url = "http://localhost:11434/api/embed", # post to local Ollama server
    content_type_json(), # alerts the embedding will come from JSON 
    body = toJSON(list(
      model = "nomic-embed-text", # uses the Ollama model specified in instructions
      input = text_strings # input is the text to embed
    ), auto_unbox = TRUE) # keeps single values from being wrapped in array
  )
  result <- content(response, as = "parsed") # parses the JSON into R list
  return(unlist(result$embeddings)) # provides a vector back
}

# vec <- get_embedding("test")
# length(vec)
