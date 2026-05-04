# Script settings and resources
set.seed(123456)              # reproducibility of code
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tm)                   # text mining
library(tidyverse)            # data science tools
library(httr)                 # http calls
library(jsonlite)             # use of JSON functionality
library(textclean)            # pre-processing
library(tidytext)             # unigram tokenization pre-processing
library(textstem)             # tool for lemmatization
library(RWeka)                # n-gram DTM # OpenJDK is an open-source solution for these java invoking libraries via University software support
library(parallel)             # parallelization
library(doParallel)           # parallelization
library(caret)                # machine learning
library(stm)                  # structural topic modeling

# Parallelizations
n_cores <- max(1L, detectCores(logical = FALSE) - 1L)  # leave 1 core free
cat("Using", n_cores, "cores\n")

# Create Stratified Sample
sample_n <- 50000 # I set this sample at 

# Data Import and Cleaning
import_tbl <- read_csv("../data/glassdoor_reviews.csv") # imports once; read_csv is slower but fast enough for purposes

glassdoor_tbl <- import_tbl %>%  # import dataset for tidyverse, small enough to import relatively quickly
  mutate(overall_rating = as.integer(overall_rating)) %>%  # convert outcome --> `overall_rating` to integer
  select(overall_rating, headline, pros, cons) %>%  # retain outcome and text columns
  filter(!is.na(overall_rating)) %>%  # drop where the outcome is missing
  mutate(
    full_review = paste(
      replace_na(headline, ""),
      replace_na(pros, ""),
      replace_na(cons, ""),
      sep = " "
    ),
    full_review = str_squish(full_review),
    doc_id = row_number()
  ) %>% 
  select(-headline, -pros, -cons) 

model_tbl <- glassdoor_tbl

# Stratified sample (preserves rating distribution)
if (nrow(model_tbl) > sample_n) {
  model_tbl <- model_tbl %>%
    group_by(overall_rating) %>%
    slice_sample(prop = sample_n / nrow(glassdoor_tbl)) %>%
    ungroup()
  cat("Sampled down to", nrow(model_tbl), "rows\n")
} #%>% 
#   write_rds("../out/data.RDS") # saves final dataset as per line 3.3

cat("Rating distribution:\n")
print(table(model_tbl$overall_rating))


# Create Holdout and Training Datasets
holdout_indices <- createDataPartition(model_tbl$doc_id, 
                                       p = .25, 
                                       list=F) # This line creates a 25/75 split of holdout:training data
model_holdout <- model_tbl[holdout_indices,] # holdout data
model_training <- model_tbl[-holdout_indices,] # training data
 

# Begin Parallelization
# local_cluster <- makeCluster(7) # using `detectCores()` I identified 8 cores, subtracting 1, I began the local cluster for parallelization 
# registerDoParallel(local_cluster) # activate cluster

# End Parallelization
# stopCluster(local_cluster)
# registerDoSEQ()

# Step 1: Data Wrangling & Pre-Processing
unigrams <- model_tbl %>%
  select(doc_id, text = full_review) %>%
  mutate(text = str_to_lower(text)) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word") %>%
  mutate(word = lemmatize_words(word))


# bigrams <- model_tbl %>%
#   select(doc_id, text = full_review) %>%
#   mutate(text = str_to_lower(text)) %>%
#   unnest_tokens(word, text, token = "ngrams", n = 2)

# Step 2: Create a Dataset with NGram tokenization
top_words <- unigrams %>% 
  count(word, sort = TRUE) %>% 
  filter(n >= 40) %>% 
  slice_head(n = 3000)

tidy_tokens <- unigrams %>% 
  semi_join(top_words, by = "word")

doc_sizes <- tidy_tokens %>%
  count(doc_id)

large_docs <- doc_sizes %>%
  filter(n > quantile(n, 0.90)) %>%
  pull(doc_id)

tidy_tokens <- tidy_tokens %>%
  filter(!doc_id %in% large_docs)

tidy_tokens <- tidy_tokens %>%
  anti_join(
    tidy_tokens %>%
      count(word) %>%
      filter(n > quantile(n, 0.99)),
    by = "word"
  )

# Step #3: Analysis

dfm <- tidy_tokens %>%
  count(doc_id, word) %>%
  cast_dfm(doc_id, word, n)

dfm <- quanteda::dfm_trim(
  dfm,
  min_termfreq = 50,   
  min_docfreq = 10
)


## Topic analysis 
stm_input <- quanteda::convert(dfm, to = "stm")
kresult <- searchK(
  stm_input$documents,
  stm_input$vocab,
  K = c(5, 10, 15),
  heldout = F,
  cores = n_cores # conducts core parralelization for optimal performance
)
plot(kresult)
topic_model <- stm(documents = stm_input$documents, 
                   vocab = stm_input$vocab, 
                   K = 10,
                   data = stm_input$meta,
                   init.type = "Spectral")

## Interpretation of topic analysis
labelTopics(topic_model, n=10)
# findThoughts(topic_model, texts=model_tbl$full_review, n=3)
plot(topic_model, type="summary", n=5)
topicCorr(topic_model)
plot(topicCorr(topic_model))

# Publication

# RQ1. Does the use of embeddings (using the nomic-embed-text LLM embeddings model) improve prediction of satisfaction beyond a rigorous tokenization strategy?
# RQ2. Does the use of topics improve prediction of satisfaction beyond a rigorous tokenization strategy?
# RQ3. Does the use of embeddings plus topics improve prediction of satisfaction beyond either alone?
# RQ4. What is the best prediction of overall job satisfaction achievable using text reviews as source data?

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


# vec <- get_embedding("test") # test my embedding function
# length(vec) # length is 768 from Ollama

# save.image(file = "../out/final.RData") #saves an .RData file as per line 3.4