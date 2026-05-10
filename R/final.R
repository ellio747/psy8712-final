# Script settings and resources
set.seed(123456)              # reproducibility of code, important for this to go at the very top for entire code pipeline reproducibility
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)            # data science tools
library(tm)                   # text mining
library(httr)                 # http calls
library(jsonlite)             # use of JSON functionality
library(tidytext)             # unigram tokenization pre-processing
library(textstem)             # tool for lemmatization
library(RWeka)                # n-gram DTM # OpenJDK is an open-source solution for these java invoking libraries via University software support
library(stm)                  # structural topic modeling
library(parallel)             # parallelization
library(doParallel)           # parallelization
library(caret)                # machine learning
library(glmnet)               # machine learning model #1
library(ranger)               # machine learning model #2
# remotes::install_version("xgboost", version = "1.7.8.1", repos = "https://cran.r-project.org") 
library(xgboost)              # machine learning model #3

# Data Import and Cleaning
import_tbl <- read_csv("../data/glassdoor_reviews.csv") # imports once; read_csv is slower but fast enough for purposes
# note the file is in my gitignore due to its presence behind the kaggle user/password infrastructure

## Creation of tibble to hold my glassdoor reviews
glassdoor_tbl <- import_tbl %>%  # import dataset for tidyverse, small enough to import relatively quickly
  mutate(overall_rating = as.integer(overall_rating)) %>%  # convert outcome --> `overall_rating` to integer
  select(overall_rating, headline, pros, cons) %>%  # retain outcome and text columns
  filter(!is.na(overall_rating)) %>%  # drop where the outcome is missing, which is an essential requirement of NLP
  mutate( # this creates a full_review variable combining the pro, con, and headlines into a single string
    full_review = str_c(headline, pros, cons, sep = " ", na.rm = TRUE), # this combine function using stringr is more efficient than paste0 and associated with our block on text processing
    full_review = str_squish(full_review), # removes whitespace, chosen to stay within tidyverse/stringr functions
    doc_id = row_number() # assigns a parent document id, which is easier than a seq() type of function
  ) %>% 
  select(-headline, -pros, -cons) # removes unnecessary sub-text columns


## Create Stratified Sample
sample_n <- 5000 # I set this sample at 5K to minimize the processing requirements on an 8 core unit and obtain robust results

# Stratified sample (preserves rating distribution) for model creation
model_tbl <- glassdoor_tbl %>%
  group_by(overall_rating) %>% # groups by overall_ratings to obtain a distribution of the baseline overall_ratings
  slice_sample(prop = sample_n / nrow(glassdoor_tbl)) %>% # slice sample takes a representative amount from the distribution at random, which is easier and more efficient than sample()
  ungroup() # ungroups by overall_rating and places them back in the tibble

# Create Holdout and Training Datasets
holdout_indices <- createDataPartition(model_tbl$overall_rating, p = .25, list=F) # This line creates a 25/75 split of holdout:training data; use of caret function that internally handles stratified splitting
model_holdout <- model_tbl[holdout_indices,]    # holdout data
model_training <- model_tbl[-holdout_indices,]  # training data

# Analysis (Step 1 - Tokenization of NLP Dataset)

## Following 4 steps of Natural Language Processing (See 8712 Week 12 Slides for steps)
## Step 1: Data Wrangling/Munging
corpus <- VCorpus(VectorSource(model_tbl$full_review)) # creation of volitle corpus; keeps documents in memory rather than on disk, which is preferred over Corpus() due to its higher disk-based storage requirement

## Step 2: Pre-Processing
corpus_prep <- corpus %>%
  tm_map(content_transformer(str_to_lower)) %>% # lowercase; str_to_lower for consistent tidyverse work; need content_transformer here as stringr functions are combined with tm-based functions
  tm_map(removeNumbers) %>% # removes numbers, as provide no useful sentiment
  tm_map(removePunctuation) %>% # removes punctuation, which is easier than regex applications
  tm_map(removeWords, stopwords("en")) %>% # removes English stopwords 
  tm_map(stripWhitespace) # removes the unnecessary whitespace created when combining words
#mutate(word = lemmatize_words(word)) # lemmatization is an incredibly slow pre-processing step; hold this out at current level of compute 

## Catuion --> must retain doc_id ordering when placing into corpus
corpus_doc_ids <- model_tbl$doc_id

## Step 3: Generate a Dataset with a "rigorous tokenization" strategy = bigrams
myTokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), 1:2), paste, collapse = " ")) # use of RWeka tokenization; ensure have java or OpenJDK installed here
}

### Build Sparse term-document matrix & Remove spares terms
glassdoor_dtm <- DocumentTermMatrix( # DocumentTermMatrix builds a sparse term-document matrix from the corpus
  corpus_prep, # inputs the corpus
  control = list(
    tokenize = myTokenizer, # uses myTokenizer function built using RWeka
    bounds = list(global = c(20, Inf)) # bounds argument drops terms appearing in fewer than 20 documents
  )
)

glassdoor_slim_dtm <- removeSparseTerms(glassdoor_dtm, .95) # removeSparseTerms further trims infrequent terms; retains terms appearing in at least 5% of documents, which was based on an efficiency for compute, although a higher n:K ratio

# Ratio of docs (N) to terms (k) = 3.16 N/k ratio
nrow(model_tbl) / length(glassdoor_slim_dtm$dimnames$Terms) # this results in a somewhat large N/k ratio; would normally aim for 2:1-3:1 but with compute at this level, retain the current ratio for model simplicity

# Convert DTM to STM format using slam
dfm2stm <- readCorpus(glassdoor_slim_dtm, type = "slam") # interprets the sparse dtm and converts into a corpus representation for use in stm input analysis, using slam package for compatibility with tm-based corpora

### Obtain optimal k, using searchK for diagnostic model
kresult <- searchK(
  dfm2stm$documents, # pulls documents from dtm
  dfm2stm$vocab, # pulls vocabulary/words from dtm
  K = seq(4,8,1) # sequences between 4 and 8 by 1, so examines 4, 5, 6, 7, and 8 k solutions
)
plot(kresult) # these plots show a consistent elbow-type solution at 5 topics

### Fit the Latent Dirichlet Allocation Structural Topic Model (STM)
topic_model <- stm(
  documents = dfm2stm$documents, # pulls documents from dtm
  vocab = dfm2stm$vocab, # pulls vocabulary/words from dtm
  K = 5, # 5 topics from the results of the kresult plot
  init.type = "LDA" # selected a Latent Dirichlet Allocation as in class
)

# Step 4: Analysis of NLP Dataset
## Interpretation of topic analysis using various analytic results
labelTopics(topic_model, n=10) # prints 10 words from each topic and examined FREX for labeling
plot(topic_model, type="summary", n=5) # plots, summarizes the topic models
topicCorr(topic_model) # provides intercorrelations of topic models, evaluating correlation strength of topics
plot(topicCorr(topic_model)) # this is an output that shows the graphical relationship between topics

## These four lines designate the document indices that have been retained & dropped following topic modeling
kept_positions <- as.integer(names(dfm2stm$documents))
kept_indices <- corpus_doc_ids[kept_positions]
all_indices <- model_tbl$doc_id
dropped_indices <- setdiff(all_indices, kept_indices)

## Assigns topic labels as user input to the kresult obtained topics, place in tibble for easy access
topic_labels <- tibble(
  topic = 1:5, # k = 5 model solution
  topic_label = c("Workplace and Management Support", # these five topics were obtained through judgment-based categorization, using FREX labeling values
                  "Work-life Balance",
                  "Lead, Growth, Company Culture",
                  "Benefits, Pay, and Work Conditions",
                  "Career Development, Organizational Reputation")
)

## This tibble coding provides for the topics and all the retained tokens
topics_tbl <- tibble(
  doc_id = kept_indices, # the retained doc_id from the indices retained after topic modeling
  topic = apply(topic_model$theta, 1, which.max), # retains the topic with the highest probability
  probability = apply(topic_model$theta, 1, max) # prints the word-topic probability association based on the LDA
) %>%
  left_join(topic_labels, by = "topic") # here the topic labels are joined on

# Analysis (Step 2 - Obtain Nomic Embeddings)

## NOTE: ensure ollama serve is running in a Terminal window in order to execute the dependencies of the embedding model
## This code was adapted from the code provided in the final assignment prompt
get_embedding <- function(text_strings) { #returns embedding vector for any string
  response <- httr::POST(
    url = "http://localhost:11434/api/embed", # post to local Ollama server
    httr::content_type_json(), # alerts the embedding will come from JSON
    body = jsonlite::toJSON(list(
      model = "nomic-embed-text", # uses the Ollama model specified in instructions
      input = as.character(text_strings) # input is the text to embed
    ), auto_unbox = TRUE) # keeps single values from being wrapped in array
  )
  result <- content(response, as = "parsed") # this pulls the parsed embedding content into a result storage area
  if (!is.null(result$embeddings)) return(result$embeddings) # examines if the embedding result is null, then it returns the result
}

# Establish a batch process to pull from the ollama server due to size; this was advised in an AI solution as pulling from a local server at a high rate may cause timeouts and errors
batch_size <- 64 # Claude recommended batch size
temp_test <- model_tbl %>%
  filter(doc_id %in% kept_indices)
texts_all_test  <- temp_test$full_review 
n_docs     <- length(texts_all_test)

# This variable produces a vector of document indices into equal sized chunks for processing the Ollama batches
batch_indices <- split(seq_len(n_docs),
                       ceiling(seq_len(n_docs) / batch_size))

# This sequence of foreach loop was a claude generated call that I evaluated several times to iterate in order to get the desired outcomes of the 768 embeddings from Ollama. 
embeddings_list <- foreach(idx = batch_indices, # creates a foreach loop iterating over the batches in order to pull them all together, idx is the index number for each batch
                           .packages = c("httr", "jsonlite"), # ensures access to the required libraries on the server
                           .combine = "c") %do% {  # %do% used rather than %dopar% because Ollama cannot handle concurrent connections
                             batch <- texts_all_test[idx] # this subsets each current batch's index
                             for (i in 1:3) { # this retry loop will attempt to batch the embedding up to three times before giving up
                               res <- try(get_embedding(batch), silent = TRUE) # tries using an HTTP request to Ollama
                               if (!inherits(res, "try-error")) break # evaluates if the res is a try error object, if it is a break will occur, if an error is thrown, a system sleep will try again
                               Sys.sleep(1) # wait a second before next trial
                             }
                             if (inherits(res, "try-error")) stop("Batch failed after retries") # this if then statement stops after three failed attempts
                             res # returns the res, or the successfully retrieved embedding
                           }


# Obtain Embedding Matrix
embed_tbl <- do.call(rbind, embeddings_list) %>%
  as_tibble(.name_repair = "unique") %>% # .name_repair prevents the unique column name warning I was dealing with
  set_names(paste0("e_", seq_len(ncol(.)))) %>% # seq_len(ncol(.)) preferred over seq_along for matrix column indexing
  mutate(
    doc_id = temp_test$doc_id, # rejoins doc_ids by position in order to preserve the batch ordering
    across(starts_with("e_"), as.numeric), # coerces list-columns to numeric vectors
    .before = 1 # places doc_id as first column for readability
  )

# Create a document term tibble
dtm_kept_indices <- corpus_doc_ids[as.integer(glassdoor_slim_dtm$dimnames$Docs)]

# Create a dtm tibble from the glassdoor_slim tibble, ensuring as.matrix is used for analysis later
dtm_tbl <- glassdoor_slim_dtm %>%
  as.matrix() %>%
  as_tibble(.name_repair = "unique") %>%
  set_names(paste0("w_", colnames(glassdoor_slim_dtm))) %>%
  mutate(doc_id = dtm_kept_indices, .before = 1)        # uses DTM-derived indices rather than kept_indices

# Create a theta tibble (probabilities)
theta_tbl <- as_tibble(topic_model$theta) %>%
  rename_with(~ paste0("t_", .x)) %>%
  mutate(doc_id = kept_indices, .before = 1)

# Combine tibbles to serve as final dataset
base_tbl_save <- model_tbl %>% select(doc_id, overall_rating) %>%
  left_join(dtm_tbl, by = "doc_id") %>%  # joins tokenization features (w_ prefix)
  left_join(embed_tbl, by = "doc_id") %>%  # joins embedding features (e_ prefix)
  left_join(theta_tbl, by = "doc_id") # joins topic probability features (t_ prefix)

# write_rds(base_tbl_save, "../out/data.RDS") # saves final dataset; commented out per assignment spec (line 3.3)
# This is saved in a Google Drive due to size constraints; see readme file

# Load Saved Data from this point forward #
# base_tbl_save <- readRDS("../out/data.RDS") # loads pre-built dataset; requires model_holdout below
# ensure that all lines from # Script Settings and Resources are run before this line
# model_holdout <- base_tbl_save %>% # reconstructs holdout index from saved data
#   slice_sample(prop = .25) 

# Reconstruct feature sets from column prefixes
base_tbl <- base_tbl_save %>% select(doc_id, overall_rating) 

feat_A <- base_tbl_save %>% # tokenization features only
  select(doc_id, overall_rating, starts_with("w_")) %>% 
  na.omit() 
feat_B <- base_tbl_save %>% # embedding features only
  select(doc_id, overall_rating, starts_with("e_")) %>% 
  na.omit() 
feat_C <- base_tbl_save %>% # topic features only
  select(doc_id, overall_rating, starts_with("t_")) %>% 
  na.omit() 
feat_D <- base_tbl_save %>%  # embeddings + topics
  select(doc_id, overall_rating, starts_with("e_"), starts_with("t_")) %>% 
  na.omit() 

# split_feat: splits any feature tibble into train/holdout lists by doc_id membership
split_feat <- function(feat_tbl) {
  is_holdout <- feat_tbl$doc_id %in% model_holdout$doc_id  # logical index; %in% preferred over match() for readability
  list(train   = feat_tbl[!is_holdout, ] %>% 
         select(-doc_id), # removes doc_id as not needed for ML models
       holdout = feat_tbl[ is_holdout, ] %>% 
         select(-doc_id)) # removes doc_id as not needed for ML models
}

splits_A <- split_feat(feat_A) # these lines create the splits for each of the four features using the previous function
splits_B <- split_feat(feat_B)
splits_C <- split_feat(feat_C)
splits_D <- split_feat(feat_D)

# Define Cross-validation control
cv_control <- trainControl(method = "cv", number = 10, verboseIter = T) # produces a 10-fold cross-validation

# Begin Parallelization
n_cores <- max(1L, detectCores(logical = FALSE) - 1L)  # leave 1 core free
local_cluster <- makeCluster(n_cores)
registerDoParallel(local_cluster) # activate cluster

# What follows is four feature sets (A = Tokenization, B = Embeddings, C = Topics, D = Embeddings + Topics) of three models (#1 = Lasso/Ridge/Elastic Net, #2 = random forest, #3 = xgbTree)
# I provide comments on parameter justification for the A, Tokenization set; decisions are equivalent for all other feature sets (B-D), except where commented by exception
modA1_tm <- system.time({ # this code provides a timing function to justify decisions based on time, compute constraints
  modelA1 <- train( # caret training function
    overall_rating ~ ., # obtaining the overall_rating as desired from assignment
    splits_A$train, # uses the A feature training data
    method = "glmnet",  # method = #1 glmnet
    na.action = na.pass, # passes any NAs (which there are many in sparse dtm data)
    preProcess = c("medianImpute","center","scale","nzv"), # pre-processing includes median imputation, centering, scaling, and removing near-zero variance
    trControl = cv_control # calls to the CV control
  )
})


modA2_tm <- system.time({ # this code provides a timing function to justify decisions based on time, compute constraints
  modelA2 <- train( # caret training function
    overall_rating ~ ., # obtaining the overall_rating as desired from assignment
    splits_A$train, # uses the A feature training data
    method = "ranger",  
    na.action = na.pass, # passes any NAs (which there are many in sparse dtm data)
    tuneLength = 3, # limits hyperparameter grid to 3 value for speed, while still exploring a reasonable number of hyperparameters
    preProcess = c("medianImpute","center","scale","nzv"), # pre-processing includes median imputation, centering, scaling, and removing near-zero variance
    trControl = cv_control # calls to the CV control
  )
})

modA3_tm <- system.time({ # this code provides a timing function to justify decisions based on time, compute constraints
  modelA3 <- train(
    overall_rating ~ .,
    splits_A$train,
    method = "xgbTree",
    na.action = na.pass,
    tuneLength = 3, 
    preProcess = c("medianImpute","center","scale","nzv"),
    trControl = cv_control
  )
})

modB1_tm <- system.time({
  modelB1 <- train(
    overall_rating ~ .,
    splits_B$train, 
    method = "glmnet",  
    na.action = na.pass, 
    preProcess = c("medianImpute","center","scale","nzv", "pca"), 
    trControl = cv_control
  )
})

modB2_tm <- system.time({
  modelB2 <- train(
    overall_rating ~ ., 
    splits_B$train, 
    method = "ranger",  
    na.action = na.pass, 
    tuneLength = 3,
    preProcess = c("medianImpute","center","scale","nzv", "pca"), 
    trControl = cv_control
  )
})

modB3_tm <- system.time({
  modelB3 <- train(
    overall_rating ~ .,
    splits_B$train,
    method = "xgbTree",
    na.action = na.pass,
    tuneLength = 3,
    preProcess = c("medianImpute","center","scale","nzv", "pca"),
    trControl = cv_control
  )
})

modC1_tm <- system.time({
  modelC1 <- train(
    overall_rating ~ .,
    splits_C$train, 
    method = "glmnet",  
    na.action = na.pass, 
    preProcess = c("medianImpute","center","scale","nzv"), 
    trControl = cv_control
  )
})


modC2_tm <- system.time({
  modelC2 <- train(
    overall_rating ~ ., 
    splits_C$train, 
    method = "ranger",  
    na.action = na.pass,
    tuneLength = 3,
    preProcess = c("medianImpute","center","scale","nzv"), 
    trControl = cv_control
  )
})

modC3_tm <- system.time({
  modelC3 <- train(
    overall_rating ~ .,
    splits_C$train,
    method = "xgbTree",
    na.action = na.pass,
    tuneLength = 3,
    preProcess = c("medianImpute","center","scale","nzv"),
    trControl = cv_control
  )
})

modD1_tm <- system.time({
  modelD1 <- train(
    overall_rating ~ .,
    splits_D$train, 
    method = "glmnet",  
    na.action = na.pass, 
    preProcess = c("medianImpute","center","scale","nzv", "pca"), 
    trControl = cv_control
  )
})


modD2_tm <- system.time({
  modelD2 <- train(
    overall_rating ~ ., 
    splits_D$train, 
    method = "ranger",
    tuneLength = 3,
    na.action = na.pass, 
    preProcess = c("medianImpute","center","scale","nzv", "pca"), 
    trControl = cv_control
  )
})

modD3_tm <- system.time({
  modelD3 <- train(
    overall_rating ~ .,
    splits_D$train,
    method = "xgbTree",
    tuneLength = 3, 
    na.action = na.pass,
    preProcess = c("medianImpute","center","scale","nzv", "pca"),
    trControl = cv_control
  )
})


# End Parallelization
stopCluster(local_cluster) #stops the cluster
registerDoSEQ() # closes out parallelization

# Function pulling the holdout R2 as demonstrated in class
ho_rsq <- function(model, splits) {
  cor(predict(model, splits$holdout, na.action = na.pass),
      splits$holdout$overall_rating)^2
}

results_tbl <- tibble(
  algo        = c("glmnet","ranger","xgbTree",
                  "glmnet","ranger","xgbTree",
                  "glmnet","ranger","xgbTree",
                  "glmnet","ranger","xgbTree")
  ,
  feature_set = c(rep("Tokenization", 3), rep("Embeddings", 3),
                  rep("Topics", 3), rep("Emb+Topics", 3)),
  cv_rsq      = c( # used from previous assignments on pulling in the CV rsquared
    max(modelA1$results$Rsquared, na.rm=T), max(modelA2$results$Rsquared, na.rm=T), max(modelA3$results$Rsquared, na.rm=T), 
    max(modelB1$results$Rsquared, na.rm=T), max(modelB2$results$Rsquared, na.rm=T), max(modelB3$results$Rsquared, na.rm=T),
    max(modelC1$results$Rsquared, na.rm=T), max(modelC2$results$Rsquared, na.rm=T), max(modelC3$results$Rsquared, na.rm=T), 
    max(modelD1$results$Rsquared, na.rm=T), max(modelD2$results$Rsquared, na.rm=T), max(modelD3$results$Rsquared, na.rm=T)
  ),
  ho_rsq      = c( # used from previous assignments bringing in the holdout rsquared
    ho_rsq(modelA1, splits_A), ho_rsq(modelA2, splits_A), ho_rsq(modelA3, splits_A),
    ho_rsq(modelB1, splits_B), ho_rsq(modelB2, splits_B), ho_rsq(modelB3, splits_B),
    ho_rsq(modelC1, splits_C), ho_rsq(modelC2, splits_C), ho_rsq(modelC3, splits_C),
    ho_rsq(modelD1, splits_D), ho_rsq(modelD2, splits_D), ho_rsq(modelD3, splits_D)
  )
) %>%
  mutate(across(c(cv_rsq, ho_rsq), ~ str_remove(round(.x, 2), "^0"))) %>%  #strips leading 0 for APA reporting
  write_csv("../out/results_5k.csv") # I changed this term to store my results at different strafication levels (1K, 2K, 5K)

times_tbl <- tibble( # this tibble records the times that it took to obtain the ML results
  glmnet_time = str_remove(round(c(modA1_tm[[3]],modB1_tm[[3]],modC1_tm[[3]],modD1_tm[[3]]), 2), "^0"), #strips leading 0 for APA reporting
  ranger_time = str_remove(round(c(modA2_tm[[3]],modB2_tm[[3]],modC2_tm[[3]],modD2_tm[[3]]), 2), "^0"), #strips leading 0 for APA reporting
  xgbTree_time = str_remove(round(c(modA3_tm[[3]],modB3_tm[[3]],modC3_tm[[3]],modD3_tm[[3]]), 2), "^0"), #strips leading 0 for APA reporting
) %>% 
  write_csv("../out/times_5k.csv") # I changed this term to store my results at different strafication levels (1K, 2K, 5K)

# Publication

# I ran this code at a n = 1K, 2K, and 5K stratified samples. Total ML models script time for 1K = 12.9 minutes; for 2K = 32 minutes; for 5K = 144.5 minutes (see times_1k.csv, times_2k.csv, times_5k.csv in /out directory). 
# Increasing the stratified sample likely results in just over double the time requirements, even using parallelization using 7 cores (on an 8 core machine). Placing on the super computer would have improve performance. 
# Extending this to a sample of 838K, this script would require approximately well beyond 5K hours (200+ days) to compute.
# The 1K sample compared to the 5K sample obtained obtained an average in-sample R2 difference of .045, and out-of-sample R2 difference of .025. Ideally, I would continue to test increasing sample size until the difference of smaller- to larger-sample R2 became asymptotic (see results_1k.csv and results_5k.csv in /out directory). 
# Given these caveats and observations, I will move on to answering the research questions based on the 5K stratified sample results. 


# RQ1. Does the use of embeddings (using the nomic-embed-text LLM embeddings model) improve prediction of satisfaction beyond a rigorous tokenization strategy?
## ARQ1: Yes, across all three ML models, using embeddings nearly doubles the predictive power of a strategy using the A feature tokenization strategy alone. For glmnet deltaR2 = .16 and .18 for in- and out-sample respectively; ranger = .11 and .10, xgbTree = .10 and .11. 
# RQ2. Does the use of topics improve prediction of satisfaction beyond a rigorous tokenization strategy?
## ARQ2: No, across the three ML models, topics alone do not improve beyond a tokenization strategy. The glmnet and xgbTree models both lose .04-.06 on in-sample while ranger loses .08. Again glmnet and xgbTree lose .05-.06 on out of sample, while ranger loses .10 on out of sample.  
# RQ3. Does the use of embeddings plus topics improve prediction of satisfaction beyond either alone?
## ARQ3: There is very minimal increase in explanation of prediction based on the combination of embeddings + topics over embeddings alone. All ML models gain a very marginal R2 increase on in-sample R2 explained, but glmnet gains slightly on out-of-sample prediction. The tree-based models are either zero increases. 
## The Embeddings + topics increase significantly over topics alone. This is all due to the amount of variance explained by the embeddings model. 
# RQ4. What is the best prediction of overall job satisfaction achievable using text reviews as source data?
## ARQ4: Given the decisions I made, the best achievable prediction of job satisfaction using text reviews comes from a glmnet-based model using embeddings + topics. This resulted in an R2 of .35 and .38 for in-sample to out-of-sample respectively. Additionally, this model was able to run at 22 minutes of compute time, which was high yet, still less than the tree-based models. 

save.image(file = "../out/final_workspace.RData") #saves an .RData file as per line 3.4, using save.image as it requires the full workspace, per the assignment
# This is saved in a Google Drive due to size constraints; see readme file
