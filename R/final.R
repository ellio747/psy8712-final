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
library(tictoc)               # timing for expanding
# remotes::install_version("xgboost", version = "1.6.0.1", repos = "https://cran.r-project.org") 

# Parallelization Attributes
n_cores <- max(1L, detectCores(logical = FALSE) - 1L)  # leave 1 core free

# Create Stratified Sample
sample_n <- 1000 # I set this sample at 50K to minimize the processing requirements on an 8 core unit and obtain robust results
# note the file is in my gitignore due to its presence behind the kaggle user/password infrastructure
# also because it is a very large file

# Data Import and Cleaning
import_tbl <- read_csv("../data/glassdoor_reviews.csv") # imports once; read_csv is slower but fast enough for purposes

# Creation of tibble to hold my glassdoor reviews
glassdoor_tbl <- import_tbl %>%  # import dataset for tidyverse, small enough to import relatively quickly
  mutate(overall_rating = as.integer(overall_rating)) %>%  # convert outcome --> `overall_rating` to integer
  select(overall_rating, headline, pros, cons) %>%  # retain outcome and text columns
  filter(!is.na(overall_rating)) %>%  # drop where the outcome is missing
  mutate( # this creates a full_review variable combining the pro, con, and headlines into a single string
    full_review = paste(
      replace_na(headline, ""),
      replace_na(pros, ""),
      replace_na(cons, ""),
      sep = " "
    ),
    full_review = str_squish(full_review), # removes whitespace
    doc_id = row_number() # assigns a parent document id
  ) %>% 
  select(-headline, -pros, -cons) # removes unnecessary columns

# Renames the glassdoor starting tibble to my model tibble
model_tbl <- glassdoor_tbl 

# Stratified sample (preserves rating distribution)
if (nrow(model_tbl) > sample_n) { #examines if the number of rows in the tibble is greater than the sample size set at 50K
  model_tbl <- model_tbl %>%
    group_by(overall_rating) %>% #groups by overall_ratings to obtain a distribution of the baseline overall_ratings
    slice_sample(prop = sample_n / nrow(glassdoor_tbl)) %>% # slice sample takes a representative amount from the distribution at random
    ungroup() # ungroups by overall_rating and places them back in the tibble
}

# Create Holdout and Training Datasets
holdout_indices <- createDataPartition(model_tbl$overall_rating,
                                       p = .25,
                                       list=F) # This line creates a 25/75 split of holdout:training data
model_holdout <- model_tbl[holdout_indices,] # holdout data
model_training <- model_tbl[-holdout_indices,] # training data

# Analysis

# Creation of the topic model
# Step 1: Data Wrangling & Pre-Processing
corpus_prep <- model_tbl %>%
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
top_words <- corpus_prep %>% 
  count(word, sort = TRUE) %>% 
  filter(n >= 40) %>% 
  slice_head(n = 3000)

tidy_tokens <- corpus_prep %>% 
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

# Step 3: Analysis of model

#Convert to DTM
dtm <- tidy_tokens %>%
  count(doc_id, word) %>%
  cast_dfm(doc_id, word, n)

dtm <- quanteda::dfm_trim( # trims low represented words
  dtm,
  min_termfreq = 50,   
  min_docfreq = 10
)


## Topic analysis 
stm_input <- quanteda::convert(dtm, to = "stm")
kresult <- searchK(
  stm_input$documents,
  stm_input$vocab,
  K = c(5, 10, 15),
  heldout = F#,
  #cores = 2 # conducts core parralelization for optimal performance
)
plot(kresult)
topic_model <- stm(documents = stm_input$documents, 
                   vocab = stm_input$vocab, 
                   K = 10,
                   data = stm_input$meta,
                   init.type = "Spectral")

## Interpretation of topic analysis
labelTopics(topic_model, n=10)
plot(topic_model, type="summary", n=5)
topicCorr(topic_model)
plot(topicCorr(topic_model))

# These three lines designate the document indices that have been retained & dropped
kept_indices  <- as.integer(names(stm_input$documents))
all_indices   <- model_tbl$doc_id
dropped_indices <- setdiff(all_indices, kept_indices)

# Assigns topic lables as user input to the kernal obtained topics
topic_labels <- tibble(
  topic       = 1:10,
  topic_label = c("WFH Policies",
                  "Employee Perks",
                  "Consulting and Client Exposure",
                  "Compensation and Benefits Issues",
                  "Workplace Stessors",
                  "Career Growth Issues",
                  "Bureacratic Gridlock",
                  "Finance & Banking",
                  "Leadership and Development",
                  "Skill Development & Training")
)

# This tibble coding provides for the topics and all the retained tokens
topics_tbl <- tibble(
  doc_id      = kept_indices,
  topic       = apply(topic_model$theta, 1, which.max),
  probability = apply(topic_model$theta, 1, max)
) %>%
  left_join(topic_labels, by = "topic")



# Obtain embeddings function
# get_embedding <- function(text_strings) { #returns embedding vector for any string
#   response <- POST(
#     url = "http://localhost:11434/api/embed", # post to local Ollama server
#     content_type_json(), # alerts the embedding will come from JSON 
#     body = toJSON(list(
#       model = "nomic-embed-text", # uses the Ollama model specified in instructions
#       input = text_strings # input is the text to embed
#     ), auto_unbox = TRUE) # keeps single values from being wrapped in array
#   )
#   result <- content(response, as = "parsed")
#   return(result$embeddings)
# }

n_cores_2 <- 2
cl <- makeCluster(n_cores_2)
registerDoParallel(cl)

get_embedding <- function(text_strings) {
  response <- httr::POST(
    url = "http://localhost:11434/api/embed",
    httr::content_type_json(),
    body = jsonlite::toJSON(list(
      model = "nomic-embed-text",
      input = as.character(text_strings)
    ), auto_unbox = TRUE)
  )
  
  result <- httr::content(response, as = "parsed")
  
  if (!is.null(result$embeddings)) return(result$embeddings)
  if (!is.null(result$embedding)) return(list(result$embedding))
  
  stop("No embeddings returned")
}

batch_size <- 64
temp_test <- model_tbl %>%
  filter(doc_id %in% kept_indices)
texts_all_test  <- temp_test$full_review 
n_docs     <- length(texts_all_test)

batch_indices <- split(seq_len(n_docs),
                       ceiling(seq_len(n_docs) / batch_size))

embeddings_list <- foreach(idx = batch_indices,
                           .packages = c("httr", "jsonlite"),
                           .combine = "c") %dopar% {
                             
                             batch <- texts_all_test[idx]
                             
                             # simple retry
                             for (i in 1:3) {
                               res <- try(get_embedding(batch), silent = TRUE)
                               if (!inherits(res, "try-error")) break
                               Sys.sleep(1)
                             }
                             
                             if (inherits(res, "try-error")) {
                               stop("Batch failed after retries")
                             }
                             
                             res  # returns list of vectors
                           }


 
# # Obtain Embedding Matrix
# embeddings_list <- get_embedding(temp_test$full_review)
# 
# embed_matrix <- do.call(rbind, embeddings_list)
# colnames(embed_matrix) <- paste0("e_", seq_len(ncol(embed_matrix)))
# embed_tbl <- as_tibble(embed_matrix) %>%
#   mutate(doc_id = temp_test$doc_id, .before = 1)

embed_matrix <- do.call(rbind, embeddings_list)
colnames(embed_matrix) <- paste0("e_", seq_len(ncol(embed_matrix)))
embed_tbl <- as_tibble(embed_matrix) %>%
  mutate(doc_id = temp_test$doc_id, .before = 1)

embed_tbl_clean <- embed_tbl %>%
  mutate(across(e_1:e_768, as.numeric))

stopCluster(cl)


dtm_tbl <- tidy_tokens %>%
  count(doc_id, word) %>%
  pivot_wider(names_from = word, values_from = n,
              values_fill = 0, names_prefix = "w_") %>%
  mutate(doc_id = as.integer(doc_id))

theta_tbl <- as_tibble(topic_model$theta) %>%
  rename_with(~ paste0("t_", .x)) %>%
  mutate(doc_id = kept_indices, .before = 1)

base_tbl_save <- model_tbl %>% select(doc_id, overall_rating) %>% 
  left_join(
    dtm_tbl, by = "doc_id"
  ) %>% 
  left_join(
    embed_tbl_clean, by = "doc_id"
  ) %>% 
  left_join(
    theta_tbl, by = "doc_id"
  )  %>% 
write_rds("../out/data.RDS") # saves final dataset as per line 3.3

model_tbl <- readRDS("out/data.RDS")

base_tbl <- model_tbl %>% select(doc_id, overall_rating)  

feat_A <- base_tbl %>% left_join(dtm_tbl,   by = "doc_id") %>% na.omit()
feat_B <- base_tbl %>% left_join(embed_tbl_clean, by = "doc_id") %>% na.omit()
feat_C <- base_tbl %>% left_join(theta_tbl, by = "doc_id") %>% na.omit()
feat_D <- base_tbl %>%
  left_join(embed_tbl_clean, by = "doc_id") %>%
  left_join(theta_tbl, by = "doc_id") %>%
  na.omit()

split_feat <- function(feat_tbl) {
  is_holdout <- feat_tbl$doc_id %in% model_holdout$doc_id
  list(train   = feat_tbl[!is_holdout, ] %>% select(-doc_id) %>% as.data.frame(),
       holdout = feat_tbl[ is_holdout, ] %>% select(-doc_id) %>% as.data.frame())
}

splits_A <- split_feat(feat_A)
splits_B <- split_feat(feat_B)
splits_C <- split_feat(feat_C)
splits_D <- split_feat(feat_D)

# Define Cross-validation control
cv_control <- trainControl(method = "cv", number = 10, verboseIter = T)

# Begin Parallelization
local_cluster <- makeCluster(n_cores)
registerDoParallel(local_cluster) # activate cluster

modelA1 <- train(overall_rating ~ .,
                 splits_A$train, 
                 method = "glmnet",  
                 na.action = na.pass, 
                 preProcess = c("medianImpute","center","scale","nzv"), 
                 trControl = cv_control)


modelA2 <- train(overall_rating ~ ., 
                 splits_A$train, 
                 method = "ranger",  
                 na.action = na.pass, 
                 preProcess = c("medianImpute","center","scale","nzv"), 
                 trControl = cv_control)

# modelA3 <- train(overall_rating ~ ., 
#                  splits_A$train, 
#                  method = "xgbTree",  
#                  na.action = na.pass, 
#                  preProcess = c("medianImpute","center","scale","nzv"), 
#                  trControl = cv_control)

modelB1 <- train(overall_rating ~ .,
                 splits_B$train, 
                 method = "glmnet",  
                 na.action = na.pass, 
                 preProcess = c("medianImpute","center","scale","nzv"), 
                 trControl = cv_control)


modelB2 <- train(overall_rating ~ ., 
                 splits_B$train, 
                 method = "ranger",  
                 na.action = na.pass, 
                 preProcess = c("medianImpute","center","scale","nzv"), 
                 trControl = cv_control)

# modelB3 <- train(overall_rating ~ ., 
#                  splits_B$train, 
#                  method = "xgbTree",  
#                  na.action = na.pass, 
#                  preProcess = c("medianImpute","center","scale","nzv"), 
#                  trControl = cv_control)

modelC1 <- train(overall_rating ~ .,
                 splits_C$train, 
                 method = "glmnet",  
                 na.action = na.pass, 
                 preProcess = c("medianImpute","center","scale","nzv"), 
                 trControl = cv_control)


modelC2 <- train(overall_rating ~ ., 
                 splits_C$train, 
                 method = "ranger",  
                 na.action = na.pass, 
                 preProcess = c("medianImpute","center","scale","nzv"), 
                 trControl = cv_control)

# modelC3 <- train(overall_rating ~ ., 
#                  splits_C$train, 
#                  method = "xgbTree",  
#                  na.action = na.pass, 
#                  preProcess = c("medianImpute","center","scale","nzv"), 
#                  trControl = cv_control)

modelD1 <- train(overall_rating ~ .,
                 splits_D$train, 
                 method = "glmnet",  
                 na.action = na.pass, 
                 preProcess = c("medianImpute","center","scale","nzv"), 
                 trControl = cv_control)


modelD2 <- train(overall_rating ~ ., 
                 splits_D$train, 
                 method = "ranger",  
                 na.action = na.pass, 
                 preProcess = c("medianImpute","center","scale","nzv"), 
                 trControl = cv_control)

# modelD3 <- train(overall_rating ~ ., 
#                  splits_D$train, 
#                  method = "xgbTree",  
#                  na.action = na.pass, 
#                  preProcess = c("medianImpute","center","scale","nzv"), 
#                  trControl = cv_control)


# End Parallelization
stopCluster(local_cluster)
registerDoSEQ()

ho_rsq <- function(model, splits) {
  cor(predict(model, splits$holdout, na.action = na.pass),
      splits$holdout$overall_rating)^2
}

results_tbl <- tibble(
  algo        = c("glmnet","ranger",#"xgbTree",
                  "glmnet","ranger",#"xgbTree",
                  "glmnet","ranger",#"xgbTree",
                  "glmnet","ranger"#,"xgbTree"
                  ),
  # feature_set = c(rep("Tokenization",2), rep("Embeddings"2),
  #                 rep("Topics",2),       rep("Emb+Topics",2)),
  cv_rsq      = c(
    max(modelA1$results$Rsquared, na.rm=T), max(modelA2$results$Rsquared, na.rm=T),
    #max(modelA3$results$Rsquared, na.rm=T), 
    max(modelB1$results$Rsquared, na.rm=T), max(modelB2$results$Rsquared, na.rm=T), 
    #max(modelB3$results$Rsquared, na.rm=T),
    max(modelC1$results$Rsquared, na.rm=T), max(modelC2$results$Rsquared, na.rm=T),
    #max(modelC3$results$Rsquared, na.rm=T), 
    max(modelD1$results$Rsquared, na.rm=T), max(modelD2$results$Rsquared, na.rm=T)#, 
    #max(modelD3$results$Rsquared, na.rm=T)
  ),
  ho_rsq      = c(
    ho_rsq(modelA1, splits_A), ho_rsq(modelA2, splits_A), #ho_rsq(modelA3, splits_A),
    ho_rsq(modelB1, splits_B), ho_rsq(modelB2, splits_B), #ho_rsq(modelB3, splits_B),
    ho_rsq(modelC1, splits_C), ho_rsq(modelC2, splits_C), #ho_rsq(modelC3, splits_C),
    ho_rsq(modelD1, splits_D), ho_rsq(modelD2, splits_D)#, #ho_rsq(modelD3, splits_D)
  )
) %>%
  mutate(across(c(cv_rsq, ho_rsq), ~ str_remove(round(.x, 2), "^0")))

# Publication
# RQ1. Does the use of embeddings (using the nomic-embed-text LLM embeddings model) improve prediction of satisfaction beyond a rigorous tokenization strategy?
# RQ2. Does the use of topics improve prediction of satisfaction beyond a rigorous tokenization strategy?
# RQ3. Does the use of embeddings plus topics improve prediction of satisfaction beyond either alone?
# RQ4. What is the best prediction of overall job satisfaction achievable using text reviews as source data?


## Temp Working Section
sample_idx <- sample(nrow(splits_A$train), 500)
small_sample <- splits_A$train[sample_idx, ]

# Step 1: Force numeric
small_sample[] <- lapply(small_sample, as.numeric)

# Step 2: Remove all-NA or all-zero columns
keep_cols <- colSums(!is.na(small_sample)) > 0 & colSums(small_sample != 0, na.rm = TRUE) > 0
small_sample_clean <- small_sample[, keep_cols]

# Step 3: Verify
dim(small_sample_clean)
sum(is.na(small_sample_clean))
table(small_sample_clean$overall_rating)

# Step 4: Train
tic()
test_model <- train(overall_rating ~ .,
                    small_sample_clean,
                    method = "xgbTree",
                    trControl = trainControl(method = "cv", number = 3, allowParallel = FALSE))
(toc)




# save.image(file = "../out/final.RData") #saves an .RData file as per line 3.4