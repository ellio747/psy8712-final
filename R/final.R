# Script settings and resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)            # data science tools
library(httr)                 # http calls
library(jsonlite)             # use of JSON functionality

# Data Import and Cleaning
final_tbl <- read_csv("../data/glassdoor_reviews.csv") # import dataset

# save dataset
write_rds(final_tbl, "../out/data.RDS") # saves final dataset as per line 3.3

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

# Visualization

# Analysis

# Publication

# RQ1. Does the use of embeddings (using the nomic-embed-text LLM embeddings model) improve prediction of satisfaction beyond a rigorous tokenization strategy?
# RQ2. Does the use of topics improve prediction of satisfaction beyond a rigorous tokenization strategy?
# RQ3. Does the use of embeddings plus topics improve prediction of satisfaction beyond either alone?
# RQ4. What is the best prediction of overall job satisfaction achievable using text reviews as source data?

# save.image(file = "../out/final.RData") #saves an .RData file as per line 3.4