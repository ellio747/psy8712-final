# Script settings and resources
set.seed(123456)              # reproducibility of code
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tm)                   # text mining
library(tidyverse)            # data science tools
library(httr)                 # http calls
library(jsonlite)             # use of JSON functionality
library(tidytext)             # unigram tokenization pre-processing
library(textstem)             # tool for lemmatization
library(RWeka)                # n-gram DTM # OpenJDK is an open-source solution for these java invoking libraries via University software support
library(parallel)             # parallelization
library(doParallel)           # parallelization
library(caret)                # machine learning
library(glmnet)
library(ranger)
# remotes::install_version("xgboost", version = "1.7.8.1", repos = "https://cran.r-project.org") 
library(xgboost)
library(stm)                  # structural topic modeling


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

# Step 2: NGram tokenization
corpus_source_tbl <- corpus_prep %>%
  group_by(doc_id) %>%
  summarise(text = paste(word, collapse = " "))

# Keep doc_ids in the same order as the corpus
corpus_doc_ids <- corpus_source_tbl$doc_id

corpus_source <- corpus_source_tbl %>%
  pull(text) %>%
  VectorSource() %>%
  VCorpus()

myTokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), 1:2), paste, collapse = " "))
}

glassdoor_dtm <- DocumentTermMatrix(
  corpus_source,
  control = list(tokenize = myTokenizer)
)

# This trims sparse terms, setting sparseness to a small term
glassdoor_slim_dtm <- removeSparseTerms(glassdoor_dtm, .998)

# Ratio of docs (N) to terms (k) = 3.16 N/k ratio
nrow(model_tbl) / length(glassdoor_slim_dtm$dimnames$Terms)

# Convert DTM to STM format using slam
dfm2stm <- readCorpus(glassdoor_slim_dtm, type = "slam")

# Obtain optimal k, using searchK for diagnostic model
kresult <- searchK(
  dfm2stm$documents,
  dfm2stm$vocab,
  K = seq(3, 10, by = 1)
)
plot(kresult) # these plots show a consistent elbow-type solution at 5 topics

# Fit the Latent Dirichlet Allocation Structural Topic Model (STM)
topic_model <- stm(
  documents = dfm2stm$documents,
  vocab = dfm2stm$vocab,
  K = 5, # 5 topics from the results of the kresult plot
  init.type = "LDA" # selected a Latent Dirichlet Allocation as in class
)

# Interpretation of topic analysis using various analytic results
labelTopics(topic_model, n=10) # prints 10 words from each topic and examined FREX for labeling
plot(topic_model, type="summary", n=5) # plots, summarizes the topic models
topicCorr(topic_model) # provides intercorrelations of topic models, evaluating correlation strength of topics
plot(topicCorr(topic_model)) # this is an output that shows the graphical relationship between topics

# These three lines designate the document indices that have been retained & dropped
kept_positions  <- as.integer(names(dfm2stm$documents))
kept_indices    <- corpus_doc_ids[kept_positions]   # map position -> true doc_id
all_indices     <- model_tbl$doc_id
dropped_indices <- setdiff(all_indices, kept_indices)

# Assigns topic labels as user input to the kresult obtained topics, place in tibble for easy access
topic_labels <- tibble(
  topic       = 1:5,
  topic_label = c("Company Culture",
                  "Work-life Balance",
                  "Compensation and Benefits Issues",
                  "Career Development Issues",
                  "Team Level Management")
)

# This tibble coding provides for the topics and all the retained tokens
topics_tbl <- tibble(
  doc_id      = kept_indices,
  topic       = apply(topic_model$theta, 1, which.max),
  probability = apply(topic_model$theta, 1, max)
) %>%
  left_join(topic_labels, by = "topic")

# Obtain embeddings
n_cores_2 <- 2
cl <- makeCluster(n_cores_2)
registerDoParallel(cl)

# NOTE: ensure ollama serve is running in a Terminal window in order to execute the dependencies of the embedding model
get_embedding <- function(text_strings) { #returns embedding vector for any string
  response <- httr::POST(
    url = "http://localhost:11434/api/embed", # post to local Ollama server
    httr::content_type_json(), # alerts the embedding will come from JSON
    body = jsonlite::toJSON(list(
      model = "nomic-embed-text", # uses the Ollama model specified in instructions
      input = as.character(text_strings) # input is the text to embed
    ), auto_unbox = TRUE) # keeps single values from being wrapped in array
  )
  result <- content(response, as = "parsed")
  if (!is.null(result$embeddings)) return(result$embeddings)
  if (!is.null(result$embedding)) return(list(result$embedding))
  stop("No embeddings returned")
}

# Establish a batch process to pull from the ollama server due to size
batch_size <- 64
temp_test <- model_tbl %>%
  filter(doc_id %in% kept_indices)
texts_all_test  <- temp_test$full_review 
n_docs     <- length(texts_all_test)

batch_indices <- split(seq_len(n_docs),
                       ceiling(seq_len(n_docs) / batch_size))

embeddings_list <- foreach(idx = batch_indices,
                           .packages = c("httr", "jsonlite"),
                           .combine = "c") %do% {  # <-- %do% not %dopar%
                             batch <- texts_all_test[idx]
                             for (i in 1:3) {
                               res <- try(get_embedding(batch), silent = TRUE)
                               if (!inherits(res, "try-error")) break
                               Sys.sleep(1)
                             }
                             if (inherits(res, "try-error")) stop("Batch failed after retries")
                             res
                           }

stopCluster(cl)

 
# Obtain Embedding Matrix
embed_matrix <- do.call(rbind, embeddings_list)
colnames(embed_matrix) <- paste0("e_", seq_len(ncol(embed_matrix)))
embed_tbl <- as_tibble(embed_matrix) %>%
  mutate(doc_id = temp_test$doc_id, .before = 1)

# Create embeddings tibble
embed_tbl_clean <- embed_tbl %>%
  mutate(across(e_1:e_768, as.numeric)) # generating numeric embeddings rather than a list

# Create a document term tibble
dtm_tbl <- corpus_prep %>%
  filter(doc_id %in% kept_indices) %>% 
  count(doc_id, word) %>%
  pivot_wider(names_from = word, values_from = n,
              values_fill = 0, names_prefix = "w_") %>%
  mutate(doc_id = as.integer(doc_id))

# Create a theta tibble (probabilities)
theta_tbl <- as_tibble(topic_model$theta) %>%
  rename_with(~ paste0("t_", .x)) %>%
  mutate(doc_id = kept_indices, .before = 1)

# Combine tibbles to serve as final dataset
base_tbl_save <- model_tbl %>% select(doc_id, overall_rating) %>% 
  left_join(
    dtm_tbl, by = "doc_id"
  ) %>% 
  left_join(
    embed_tbl_clean, by = "doc_id"
  ) %>% 
  left_join(
    theta_tbl, by = "doc_id"
  )

write_rds(base_tbl_save, "../out/data.RDS") # saves final dataset as per line 3.3
# base_tbl <- readRDS("../out/data.RDS")

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


modA1_tm <- system.time({
  modelA1 <- train(
    overall_rating ~ .,
    splits_A$train, 
    method = "glmnet",  
    na.action = na.pass, 
    preProcess = c("medianImpute","center","scale","nzv"), 
    trControl = cv_control
  )
})


modA2_tm <- system.time({
  modelA2 <- train(
    overall_rating ~ ., 
    splits_A$train, 
    method = "ranger",  
    na.action = na.pass, 
    preProcess = c("medianImpute","center","scale","nzv"), 
    trControl = cv_control
  )
})

modA3_tm <- system.time({
  modelA3 <- train(overall_rating ~ .,
                   splits_A$train,
                   method = "xgbTree",
                   na.action = na.pass,
                   preProcess = c("medianImpute","center","scale","nzv"),
                   trControl = cv_control
  )
})

modB1_tm <- system.time({
  modelB1 <- train(overall_rating ~ .,
                   splits_B$train, 
                   method = "glmnet",  
                   na.action = na.pass, 
                   preProcess = c("medianImpute","center","scale","nzv"), 
                   trControl = cv_control
  )
})

modB2_tm <- system.time({
  modelB2 <- train(overall_rating ~ ., 
                   splits_B$train, 
                   method = "ranger",  
                   na.action = na.pass, 
                   preProcess = c("medianImpute","center","scale","nzv"), 
                   trControl = cv_control
  )
})

modB3_tm <- system.time({
  modelB3 <- train(overall_rating ~ .,
                   splits_B$train,
                   method = "xgbTree",
                   na.action = na.pass,
                   preProcess = c("medianImpute","center","scale","nzv"),
                   trControl = cv_control
  )
})

modC1_tm <- system.time({
  modelC1 <- train(overall_rating ~ .,
                   splits_C$train, 
                   method = "glmnet",  
                   na.action = na.pass, 
                   preProcess = c("medianImpute","center","scale","nzv"), 
                   trControl = cv_control
  )
})


modC2_tm <- system.time({
  modelC2 <- train(overall_rating ~ ., 
                   splits_C$train, 
                   method = "ranger",  
                   na.action = na.pass, 
                   preProcess = c("medianImpute","center","scale","nzv"), 
                   trControl = cv_control
  )
})

modC3_tm <- system.time({
  modelC3 <- train(overall_rating ~ .,
                   splits_C$train,
                   method = "xgbTree",
                   na.action = na.pass,
                   preProcess = c("medianImpute","center","scale","nzv"),
                   trControl = cv_control
  )
})

modD1_tm <- system.time({
  modelD1 <- train(overall_rating ~ .,
                   splits_D$train, 
                   method = "glmnet",  
                   na.action = na.pass, 
                   preProcess = c("medianImpute","center","scale","nzv"), 
                   trControl = cv_control
  )
})


modD2_tm <- system.time({
  modelD2 <- train(overall_rating ~ ., 
                   splits_D$train, 
                   method = "ranger",  
                   na.action = na.pass, 
                   preProcess = c("medianImpute","center","scale","nzv"), 
                   trControl = cv_control
  )
})

modD3_tm <- system.time({
  modelD3 <- train(overall_rating ~ .,
                   splits_D$train,
                   method = "xgbTree",
                   na.action = na.pass,
                   preProcess = c("medianImpute","center","scale","nzv"),
                   trControl = cv_control
  )
})


# End Parallelization
stopCluster(local_cluster)
registerDoSEQ()

ho_rsq <- function(model, splits) {
  cor(predict(model, splits$holdout, na.action = na.pass),
      splits$holdout$overall_rating)^2
}

results_tbl <- tibble(
  algo        = c("glmnet","ranger", "xgbTree",
                  "glmnet","ranger", "xgbTree",
                  "glmnet","ranger", "xgbTree",
                  "glmnet","ranger","xgbTree")
                  ,
  feature_set = c(rep("Tokenization", 3), rep("Embeddings", 3),
                  rep("Topics", 3), rep("Emb+Topics", 3)),
  cv_rsq      = c(
    max(modelA1$results$Rsquared, na.rm=T), max(modelA2$results$Rsquared, na.rm=T), max(modelA3$results$Rsquared, na.rm=T), 
    max(modelB1$results$Rsquared, na.rm=T), max(modelB2$results$Rsquared, na.rm=T), max(modelB3$results$Rsquared, na.rm=T),
    max(modelC1$results$Rsquared, na.rm=T), max(modelC2$results$Rsquared, na.rm=T), max(modelC3$results$Rsquared, na.rm=T), 
    max(modelD1$results$Rsquared, na.rm=T), max(modelD2$results$Rsquared, na.rm=T), max(modelD3$results$Rsquared, na.rm=T)
  ),
  ho_rsq      = c(
    ho_rsq(modelA1, splits_A), ho_rsq(modelA2, splits_A), ho_rsq(modelA3, splits_A),
    ho_rsq(modelB1, splits_B), ho_rsq(modelB2, splits_B), ho_rsq(modelB3, splits_B),
    ho_rsq(modelC1, splits_C), ho_rsq(modelC2, splits_C), ho_rsq(modelC3, splits_C),
    ho_rsq(modelD1, splits_D), ho_rsq(modelD2, splits_D), ho_rsq(modelD3, splits_D)
  )
) %>%
  mutate(across(c(cv_rsq, ho_rsq), ~ str_remove(round(.x, 2), "^0")))

times_tbl <- tibble( 
  glmnet_time = str_remove(round(c(modA1_tm[[3]],modB1_tm[[3]],modC1_tm[[3]],modD1_tm[[3]]), 2), "^0"),
  ranger_time = str_remove(round(c(modA2_tm[[3]],modB2_tm[[3]],modC2_tm[[3]],modD2_tm[[3]]), 2), "^0"),
  xgbTree_time = str_remove(round(c(modA3_tm[[3]],modB3_tm[[3]],modC3_tm[[3]],modD3_tm[[3]]), 2), "^0"),
) 

# Publication
# RQ1. Does the use of embeddings (using the nomic-embed-text LLM embeddings model) improve prediction of satisfaction beyond a rigorous tokenization strategy?
# RQ2. Does the use of topics improve prediction of satisfaction beyond a rigorous tokenization strategy?
# RQ3. Does the use of embeddings plus topics improve prediction of satisfaction beyond either alone?
# RQ4. What is the best prediction of overall job satisfaction achievable using text reviews as source data?

# save.image(file = "../out/final.RData") #saves an .RData file as per line 3.4