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

# Parallelizations
n_cores <- max(1L, detectCores(logical = FALSE) - 1L)  # leave 1 core free
cat("Using", n_cores, "cores\n")

# Create Stratified Sample
sample_n <- 50000

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
  select(-headline, -pros, -cons) %>% 
  write_rds("../out/data.RDS") # saves final dataset as per line 3.3

# Stratified sample (preserves rating distribution)
if (nrow(model_tbl) > SAMPLE_N) {
  model_tbl <- model_tbl %>%
    group_by(overall_rating) %>%
    slice_sample(prop = SAMPLE_N / nrow(model_tbl)) %>%
    ungroup()
  cat("Sampled down to", nrow(model_tbl), "rows\n")
}

cat("Rating distribution:\n")
print(table(model_tbl$overall_rating))

split     <- initial_split(model_tbl, prop = 0.80, strata = overall_rating)
train_tbl <- training(split)
test_tbl  <- testing(split)

 

# Begin Parallelization
local_cluster <- makeCluster(7) # using `detectCores()` I identified 8 cores, subtracting 1, I began the local cluster for parallelization 
registerDoParallel(local_cluster) # activate cluster

# Step 1: Data Wrangling 
corpus <- VCorpus(VectorSource(glassdoor_tbl$full_review))

# Step 2: Pre-Processing
corpus_prep <- corpus %>%
  tm_map(content_transformer(function(x)
    str_remove_all(x, "(?<=\\b[A-Z])\\.(?=[A-Z]\\b)"))) %>%
  tm_map(content_transformer(replace_contraction)) %>%
  tm_map(content_transformer(str_to_lower)) %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(content_transformer(lemmatize_words)) %>%
  tm_map(removeWords, stopwords("en")) %>%
  tm_map(stripWhitespace)

## Conversion into a DTM
# DTM <- DocumentTermMatrix(corpus_prep) 
# DTM %>% as.matrix %>% as_tibble %>% View

# Step 2: Create a Dataset with NGram tokenization
small_corpus <- corpus_prep[1:1000,]
myTokenizer <- function(x) { NGramTokenizer(x, Weka_control(min=1, max=2)) }
DTM <- DocumentTermMatrix(
  small_corpus, 
  control = list(tokenize = myTokenizer))
slimmed_dtm <- removeSparseTerms(DTM, .97)
DTM_tbl <- slimmed_dtm %>% as.matrix %>% as_tibble

# End Parallelization
stopCluster(local_cluster)
registerDoSEQ()

# Visualization

# Step #3 Analysis

## Topic analysis 
dfm2stm <- readCorpus(slimmed_dtm, type="slam")
kresult <- searchK(
  dfm2stm$documents,
  dfm2stm$vocab,
  K = seq(2, 20, by = 2)
)
plot(kresult)
topic_model <- stm(dfm2stm$documents, 
                   dfm2stm$vocab, 
                   7)

## Interpretation of topic analysis
labelTopics(topic_model, n=10)
findThoughts(topic_model, texts=blogs$documents, n=3)
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