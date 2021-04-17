---
title: "README"
author: "amochizuki"
date: "4/13/2021"
output: 
  html_document: 
    keep_md: yes
---



## Text prediction capstone project

### Data import
The data is taken from the Coursera Swiftkey dataset and was read into R using the following code with a mishmash of the `quanteda`, `tm`, and `tidytext` packages.

```r
library(shiny)
library(tm)
library(tokenizers)
library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(dplyr)
library(tibble)
library(tokenizers)
library(stringr)
library(colourpicker)

setwd("~/capstone")

if(!dir.exists("~/capstone/final")){
        tmp <- tempfile()
        download.file("https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip", destfile = tmp)
        unzip(tmp)
        unlink(tmp)
}

eng <- VCorpus(DirSource("final/en_US", encoding = "UTF-8"), 
               readerControl = list(reader = readPlain, 
                                    language = "en-US",
                                    load = FALSE))
```

### Downsampling
For the sake of speed and publication size of this README, the corpus was significantly downsampled. 


```r
downsample <- 0.001 # for better accuracy but slower speeds, increase this to 0.1 or higher

eng <- c(eng[[1]]$content, eng[[2]]$content, eng[[3]]$content)

set.seed(1234)
inTrain <- sample(1:length(eng)[1], as.integer(downsample*length(eng)))

train <- corpus(eng[inTrain])
```

### Tokenization
Tokenization is carried out with the `quanteda` package's `tokens` function, removing punctuation, symbols and numbers. Although models with stopword removal were tried, they were ultimately left in for greater accuracy, particularly for input ngrams < 3.

```r
tokens <- tokens(train,
                 remove_punct = TRUE,
                 remove_symbols = TRUE,
                 remove_numbers = TRUE) #%>%
  #tokens_remove(pattern = stopwords("english"))
```

### ngrams
The tokens are then reconstructed into 1-grams, 2-grams, 3-grams, and 4-grams, then counted with the document-feature matrix creation function `dfm` from `quanteda`.

```r
token_counter <- function(x) {
  dfm(x) %>% 
    colSums() %>%
    data.frame() %>%
    rownames_to_column("value") %>%
    `colnames<-`(c("value", "n")) %>%
    arrange(desc(n))
}

unigs <- token_counter(tokens)

bigrs <- tokens_ngrams(tokens, n = 2, concatenator = " ") %>% token_counter

trigs <- tokens_ngrams(tokens, n = 3, concatenator = " ") %>% token_counter

tetrags <- tokens_ngrams(tokens, n = 4, concatenator = " ") %>% token_counter
```

### Katz backoff model
(reference)[https://rpubs.com/salus0324/484129]
I'll be honest, I had to rely heavily on the above code for the Katz backoff model, but modified it to create a 4-gram model. 

```r
getObservedTetrag <- function(triPre=triPre, tetrags=tetrags){
    output <- tibble(ngrams=vector(mode='character', length=0), n=vector(mode='integer', length=0))
    tetragram_index <- grep(paste0("^", triPre," "), tetrags$value)
    output <- tetrags[tetragram_index,]
    return(output)
}

getObservedTetragramProb <- function(ObservedTetrag=ObservedTetrag, trigs=trigs, triPre=triPre, gamma=gamma){
    if(nrow(ObservedTetrag)<1) return(NULL)
    obCount <- trigs[which(trigs$value ==triPre),]$n
    obTetragProbs <- (ObservedTetrag$n-gamma)/obCount
    output <- cbind(as_tibble(ObservedTetrag$value), obTetragProbs)
    colnames(output) <- c("ngram", "prob")
    row.names(output) <- 1:nrow(output)
    return(output)
}

# Find words that complete tetragrams
getUnobservedTetrags <- function(ObservedTetrag=ObservedTetrag, unigs=unigs){
    observedlast <- sapply(ObservedTetrag$value, function(y) tail(strsplit(y, " ")[[1]],1))
    output <- unigs[!(unigs$value %in% observedlast),]$value
    return(output)
}

# Calculate Alpha wi-1
### Step 3.2. Calculate α(wi−1)
getAlphaTrig <- function(triPre=triPre, trigs=trigs, gamma=0.5){
    w_i_1 <- paste(strsplit(triPre, " ")[[1]][2], strsplit(triPre, " ")[[1]][3])
    w_i_1 <- bigrs[which(bigrs$value==w_i_1),]
    trigramcount <- trigs[grep(paste0("^",w_i_1$value," "), trigs$value),]
    if (nrow(trigramcount)<1) return(1)
    output <- 1- sum((trigramcount$n-gamma)/w_i_1$n)
    return(output)
}

### Step 3.3. Calculate backed off probabilities $$q_{BO} for bigrams
getBOtrigrams <- function(triPre=triPre, UnobservedTetrags=UnobservedTetrags){
    w_i_1 <- paste(strsplit(triPre, " ")[[1]][2], strsplit(triPre, " ")[[1]][3])
    output <- paste0(w_i_1, " ", UnobservedTetrags)
    return(output)
}

getObsBOTrigrams <- function(trigs=trigs, BOtrigrams=BOtrigrams){
    output <- trigs[trigs$value %in% BOtrigrams,]
    return(output)
}

getUnObsBOTrigrams <- function(trigs=trigs, BOtrigrams=BOtrigrams, ObsBOTrigrams=ObsBOTrigrams){
    output <- BOtrigrams[!(BOtrigrams %in% ObsBOTrigrams$value)]
    return(output)
}

getObsBOTrigramsProbs <- function(triPre=triPre, ObsBOTrigrams=ObsBOTrigrams, bigrs=bigrs, gamma=gamma){
    w_i_1 <- paste(strsplit(triPre, " ")[[1]][2], strsplit(triPre, " ")[[1]][3])
    w_i_1 <- bigrs[which(bigrs$value == w_i_1),]
    output <- (ObsBOTrigrams$n-gamma)/w_i_1$n 
    output <- tibble(value=ObsBOTrigrams$value, prob=output, check.rows = F) 
    return(output)
}

getUnObsBOTrigramsProbs <- function(UnObsBOTrigrams=UnObsBOTrigrams, bigrs=bigrs, AlphaTrig=AlphaTrig){
    #get the unobserved bigram tails
    UnObsBOTrigramsTails <- sapply(UnObsBOTrigrams, function(y) {
        paste(strsplit(y, " ")[[1]][2], strsplit(y, " ")[[1]][3])
    })
    UnObsBOTrigramsTails <- bigrs[bigrs$value %in% UnObsBOTrigramsTails,]
    denom <- sum(UnObsBOTrigramsTails$n)
    output <- tibble(value=UnObsBOTrigramsTails$value, prob=(AlphaTrig*UnObsBOTrigramsTails$n/denom), 
                         check.rows = F)
    return(output)
}


## Step 4. Calculated discounted probability mass that will be distributed to unobserved trigram, α(wi−2,wi−1)
### α(wi−2,wi−1)=1−∑w⊆A(wi−1)c(wi−2,wi−1,w)−γc(wi−2,wi−1)

getAlphaTetrag <- function(triPre=triPre, tetrags=tetrags, gamma=0.5){
    tetragscount <- tetrags[grep(paste0("^",triPre," "), tetrags$value),]
    trigPrecount <- trigs[which(trigs$value==triPre),]
    if (nrow(tetragscount)<1) return(1)
    output <- 1- sum((tetragscount$n-gamma)/trigPrecount$n)
    return(output)
}



##Step 5. Calculate unobserved trigram probabilities qBO(wi∣wi−2,wi−1)
##qBO(wi∣wi−2,wi−1)=α(wi−2,wi−1)qBO(wi∣wi−1)∑w⊆B(wi−2,wi−1)qBO(w∣wi−1)
getUnObsTetragramProbs <- function(triPre=triPre, QboTrigrams=QboTrigrams, AlphaTetrag=AlphaTetrag){
    sumQboTrigrams <- sum(QboTrigrams$prob)
    UnObsTetragrams <- paste(str_split(triPre, " ")[[1]][1], QboTrigrams$value, sep=" ")
    output <- AlphaTetrag*QboTrigrams$prob/sumQboTrigrams
    output <- tibble(ngram=UnObsTetragrams, prob=output)
    return(output)
}
```

### Adding keyness
The problem with the 4-gram model was that it failed to pick up on context from longer text inputs, such as for the quizzes. By taking the entire text input and tokenizing with stopword removal, `textstat_keyness` from the `quanteda` package finds associated words. The code then merges this with the Katz backoff model output and assigns each word an arbitrarily calculated score from the chi2 output of `textstat_keyness` and the probability output from the backoff model. 


```r
assoc_tokens <- function(text = text, QboTetragrams = QboTetragrams, choices = choices, tokens = tokens) {
    text_toks <- tokens(text, remove_punct = TRUE, remove_symbols = TRUE, remove_numbers = TRUE)
    text_toks <- tokens_remove(text_toks, stopwords("english"))[[1]]
    toks_inside <- tokens_keep(tokens, pattern = text_toks, window = 10)
    toks_inside <- tokens_remove(toks_inside, pattern = text_toks) # remove the keywords
    toks_outside <- tokens_remove(tokens, pattern = text_toks, window = 10)
    dfmat_inside <- dfm(toks_inside)
    dfmat_outside <- dfm(toks_outside)
    tstat_key_inside <- textstat_keyness(rbind(dfmat_inside, dfmat_outside), 
                                         target = seq_len(ndoc(dfmat_inside)))
    df <- tstat_key_inside %>%
        full_join(QboTetragrams, by = "feature") %>%
        mutate(prob = ifelse(is.na(prob), 0, prob)) %>%
        mutate(chi2 = ifelse(is.na(chi2), 0, chi2)) %>%
        mutate(score = log(abs(1+chi2))*(1+prob*100)) %>%
        arrange(desc(score))
    ifelse(is.null(choices), 
           output <- head(df, 5), 
           output <- df %>% filter(feature %in% choices))
    return(output)
}

getNextWord <- function(ObservedTetragramProb=ObservedTetragramProb, UnObsTetragramProb=UnObsTetragramProb, text=text, choices=choices, tokens=tokens){
    QboTetragrams <- rbind(ObservedTetragramProb, UnObsTetragramProb)
    QboTetragrams <- QboTetragrams[order(-QboTetragrams$prob),]
    QboTetragrams$ngram <- sapply(QboTetragrams$ngram, function(y) tail(strsplit(y, " ")[[1]],1))
    QboTetragrams$ngram <- as.character(QboTetragrams$ngram)
    colnames(QboTetragrams) <- c("feature", "prob")
    text_length <- length(tokenize_ngrams(text, n = 1, simplify = TRUE))
    ifelse(text_length<4 & is.null(choices), 
           output <- head(QboTetragrams, 5), 
           ifelse(text_length<4 & !is.null(choices),
                  output <- QboTetragrams %>% filter(feature %in% choices) %>% arrange(desc(prob)),
                  output <- assoc_tokens(text, QboTetragrams, choices, tokens)))
    return(output)
}
    

predictNextWord <- function(gamma, text, choices=NULL, unigs, bigrs, trigs, tetrags, tokens){
    triPre <- tokenize_ngrams(text, n = 3, simplify = TRUE) %>% tail(1)
    ObservedTetrag <- getObservedTetrag(triPre, tetrags)
    ObservedTetragramProb <- getObservedTetragramProb(ObservedTetrag, trigs, triPre,gamma)
    UnobservedTetrags <- getUnobservedTetrags(ObservedTetrag, unigs)
    AlphaTrig <- getAlphaTrig(triPre, trigs, gamma)
    BOtrigrams <- getBOtrigrams(triPre,UnobservedTetrags)
    ObsBOTrigrams <- getObsBOTrigrams(trigs, BOtrigrams)
    UnObsBOTrigrams <- getUnObsBOTrigrams(trigs=trigs, BOtrigrams=BOtrigrams, ObsBOTrigrams=ObsBOTrigrams)
    ObsBOTrigramsProbs <- getObsBOTrigramsProbs(triPre, ObsBOTrigrams, bigrs, gamma)
    UnObsBOTrigramsProbs <- getUnObsBOTrigramsProbs(UnObsBOTrigrams=UnObsBOTrigrams, bigrs=bigrs, AlphaTrig=AlphaTrig)
    QboTrigrams <- rbind(ObsBOTrigramsProbs, UnObsBOTrigramsProbs)
    AlphaTetrag <- getAlphaTetrag(triPre, tetrags, gamma)
    UnObsTetragramProbs <- getUnObsTetragramProbs(triPre, QboTrigrams, AlphaTetrag)
    output <- getNextWord(ObservedTetragramProb, UnObsTetragramProbs, text, choices, tokens)
    return(output)
} 
```

### Falling back to a 3-gram model
However, the merged model still tended to fail with smaller text inputs. As such, a 3-gram model was added in case the above output had zero hits. This is also the point where I realized that leaving in the stopwords was helpful (especially if the input character string was all stopwords). 


```r
getObservedTrig <- function(bigPre=bigPre, trigs=trigs){
        output <- tibble(ngrams=vector(mode='character', length=0), n=vector(mode='integer', length=0))
        trigram_index <- grep(paste0("^", bigPre," "), trigs$value)
        output <- trigs[trigram_index,]
        return(output)
}

getObservedTrigramProb <- function(ObservedTrig=ObservedTrig, bigrs=bigrs, bigPre=bigPre, gamma=gamma){
        if(nrow(ObservedTrig)<1) return(NULL)
        obCount <- bigrs[which(bigrs$value ==bigPre),]$n
        obTrigProbs <- (ObservedTrig$n-gamma)/obCount
        output <- cbind(as_tibble(ObservedTrig$value), obTrigProbs)
        colnames(output) <- c("ngram", "prob")
        row.names(output) <- 1:nrow(output)
        return(output)
}

# Find words that complete unobserved trigrams
getUnobservedTrigs <- function(ObservedTrig=ObservedTrig, unigs=unigs){
        observedlast <- sapply(ObservedTrig$value, function(y) tail(strsplit(as.character(y), " ")[[1]],1))
        output <- unigs[!(unigs$value %in% observedlast),]$value
        return(output)
}

getAlphaBigr <- function(bigPre=bigPre, bigrs=bigrs, gamma=0.5){
        w_i_1 <- strsplit(bigPre, " ")[[1]][2]
        w_i_1 <- unigs[which(unigs$value==w_i_1),]
        bigramcount <- bigrs[grep(paste0("^",w_i_1$value," "), bigrs$value),]
        if (nrow(bigramcount)<1) return(1)
        output <- 1- sum((bigramcount$n-gamma)/w_i_1$n)
        return(output)
}

getBObigrams <- function(bigPre=bigPre, UnobservedTrigs=UnobservedTrigs){
        w_i_1 <- strsplit(bigPre, " ")[[1]][2]
        output <- paste0(w_i_1, " ", UnobservedTrigs)
        return(output)
}

getObsBOBigrams <- function(bigrs=bigrs, BObigrams=BObigrams){
        output <- bigrs[which(bigrs$value %in% BObigrams),]
        return(output)
}

getUnObsBOBigrams <- function(bigrs=bigrs, BObigrams=BObigrams, ObsBOBigrams=ObsBOBigrams){
        output <- BObigrams[!(BObigrams %in% ObsBOBigrams$value)]
        return(output)
}

getObsBOBigramsProbs <- function(bigPre=bigPre, ObsBOBigrams=ObsBOBigrams, unigs=unigs, gamma=gamma){
        w_i_1 <- strsplit(bigPre, " ")[[1]][2]
        w_i_1 <- unigs[which(unigs$value == w_i_1),]
        output <- (ObsBOBigrams$n-gamma)/w_i_1$n
        output <- tibble(value=ObsBOBigrams$value, prob=output)
        return(output)
}

getUnObsBOBigramsProbs <- function(UnObsBOBigrams=UnObsBOBigrams, unigs=unigs, AlphaBigr=AlphaBigr){
        #get the unobserved bigram tails
        UnObsBOBigramsTails <- sapply(UnObsBOBigrams, function(y) tail(strsplit(y, " ")[[1]],1))
        UnObsBOBigramsTails <- unigs[unigs$value %in% UnObsBOBigramsTails,]
        denom <- sum(UnObsBOBigramsTails$n)
        output <- tibble(value=UnObsBOBigramsTails$value, prob=(AlphaBigr*UnObsBOBigramsTails$n/denom))
        return(output)
}

getAlphaTrigram <- function(bigPre=bigPre, trigs=trigs, gamma=0.5){
        trigscount <- trigs[grep(paste0("^",bigPre," "), trigs$value),]
        bigPrecount <- bigrs[which(bigrs$value==bigPre),]
        if (nrow(trigscount)<1) return(1)
        output <- 1- sum((trigscount$n-gamma)/bigPrecount$n)
        return(output)
}

getUnObsTrigramProbs <- function(bigPre=bigPre, QboBigrams=QboBigrams, AlphaTrig=AlphaTrig){
        sumQboBigrams <- sum(QboBigrams$prob)
        UnObsTrigrams <- paste(str_split(bigPre, " ")[[1]][1], QboBigrams$value, sep=" ")
        output <- AlphaTrig*QboBigrams$prob/sumQboBigrams
        output <- tibble(ngram=UnObsTrigrams, prob=output)
        return(output)
}

assoc_tokens_trigrams <- function(text = text, QboTrigrams = QboTrigrams, choices = choices, tokens = tokens) {
        text_toks <- tokens(text, remove_punct = TRUE, remove_symbols = TRUE, remove_numbers = TRUE)
        text_toks <- tokens_remove(text_toks, stopwords("english"))[[1]]
        toks_inside <- tokens_keep(tokens, pattern = text_toks, window = 10)
        toks_inside <- tokens_remove(toks_inside, pattern = text_toks) # remove the keywords
        toks_outside <- tokens_remove(tokens, pattern = text_toks, window = 10)
        dfmat_inside <- dfm(toks_inside)
        dfmat_outside <- dfm(toks_outside)
        tstat_key_inside <- textstat_keyness(rbind(dfmat_inside, dfmat_outside), 
                                             target = seq_len(ndoc(dfmat_inside)))
        df <- tstat_key_inside %>%
                full_join(QboTrigrams, by = "feature") %>%
                mutate(prob = ifelse(is.na(prob), 0, prob)) %>%
                mutate(chi2 = ifelse(is.nan(chi2), 0, chi2)) %>%
                mutate(score = log(abs(1.1+chi2))*(1+prob*100)) %>%
                arrange(desc(score))
        ifelse(is.na(choices), 
               output <- head(df, 5), 
               output <- df %>% filter(feature %in% choices))
        return(output)
}

getNextWord_trigrams <- function(ObservedTrigramProb=ObservedTrigramProb, UnObsTrigramProbs=UnObsTrigramProbs, text=text, choices=choices, tokens = tokens){
        QboTrigrams <- rbind(ObservedTrigramProb, UnObsTrigramProbs)
        QboTrigrams <- QboTrigrams[order(-QboTrigrams$prob),]
        QboTrigrams$ngram <- sapply(QboTrigrams$ngram, function(y) tail(strsplit(y, " ")[[1]],1))
        colnames(QboTrigrams) <- c("feature", "prob")
        text_length <- length(tokenize_ngrams(text, n = 1, simplify = TRUE))
        ifelse(text_length<3 & is.na(choices), 
               output <- head(QboTrigrams, 5), 
               ifelse(text_length<3 & !is.na(choices),
                      output <- QboTrigrams %>% filter(feature %in% choices) %>% arrange(desc(prob)),
                      output <- assoc_tokens_trigrams(text, QboTrigrams, choices, tokens)))
        return(output)
}

predictNextWord_trigrams <- function(gamma, text, choices, unigs, bigrs, trigs, tokens){
        bigPre <- tokenize_ngrams(text, n = 2, simplify = TRUE) %>% tail(1)
        ObservedTrig <- getObservedTrig(bigPre, trigs)
        ObservedTrigramProb <- getObservedTrigramProb(ObservedTrig, bigrs, bigPre,gamma)
        UnobservedTrigs <- getUnobservedTrigs(ObservedTrig, unigs)
        AlphaBigr <- getAlphaBigr(bigPre, bigrs, gamma)
        BObigrams <- getBObigrams(bigPre,UnobservedTrigs)
        ObsBOBigrams <- getObsBOBigrams(bigrs, BObigrams)
        UnObsBOBigrams <- getUnObsBOBigrams(bigrs=bigrs, BObigrams=BObigrams, ObsBOBigrams=ObsBOBigrams)
        ObsBOBigramsProbs <- getObsBOBigramsProbs(bigPre, ObsBOBigrams, unigs, gamma)
        UnObsBOBigramsProbs <- getUnObsBOBigramsProbs(UnObsBOBigrams=UnObsBOBigrams, unigs=unigs, AlphaBigr=AlphaBigr)
        QboBigrams <- rbind(ObsBOBigramsProbs, UnObsBOBigramsProbs)
        AlphaTrigram <- getAlphaTrigram(bigPre, trigs, gamma)
        UnObsTrigramProbs <- getUnObsTrigramProbs(bigPre, QboBigrams, AlphaTrigram)
        output <- getNextWord_trigrams(ObservedTrigramProb, UnObsTrigramProbs, text, choices, tokens)
        return(output)
} 
```

### Testing
On a 3-year-old laptop with 8GB of RAM, the model can spit out predictions in about 12 seconds on a severely downsampled model. Testing is done on a portion of the corpus that is not used in building the model. The accuracy was in the 90ish% range with the below code on a 10% downsample, although it drops to the 60s when downsampling to the size needed to run on shinyapps.io. Here again, the size of the corpus was further reduced for the purposes of making this README. 


```r
gamma <- 0.5
text <- "I'd just like all of these questions answered, a presentation of evidence, and a jury to settle the"
choices <- c("matter", "account", "case", "incident")
# speed
start_time <- Sys.time()
NextWord <- predictNextWord(gamma, text, choices, unigs, bigrs, trigs, tetrags, tokens)
end_time <- Sys.time()
round(end_time - start_time, 2)
```

```
## Time difference of 2.59 secs
```

```r
# accuracy
test <- eng[-inTrain]
set.seed(1234)
inTest <- sample(1:length(test)[1], as.integer(0.1*length(test)))
test <- corpus(test[inTest])

tetrags_test <- test %>% tokenize_ngrams(n = 4) %>% unlist %>% as_tibble %>% count(value, sort = TRUE)

set.seed(1234)
SampleWords<-sample(unigs$value[1:100], 4)

TestTetragrams <- tetrags_test[(sapply(tetrags_test$value, function(y) tail(strsplit(y, " ")[[1]],1)) %in% SampleWords),]
set.seed(1234)
SampleTetrags <- sample(TestTetragrams$value, 100)

TriPreList<- sapply(SampleTetrags, function(y) paste(head(strsplit(y, " ")[[1]], 3), collapse = " "))

prediction <- sapply(TriPreList, function(y) predictNextWord(gamma, y, SampleWords, unigs, bigrs, trigs, tetrags, tokens)$feature[1])
sum(paste0(TriPreList,' ', prediction)==SampleTetrags)/length(SampleTetrags)
```

```
## [1] 0.41
```
### Conclusions
The model is much less accurate when downsampling to the sizes needed (1% of the full corpus) to run within the memory limits of a free shinyapps account. There are probably faster and more accurate models out there, but I was already too far down this rabbithole to start over. The Shiny app itself is pretty boring, so I added in the `colourpicker` package to spice things up, as well as a reset button to keep it from running if additional inputs are entered. 
