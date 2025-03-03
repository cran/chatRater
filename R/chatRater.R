#' @title Multi-Model LLM API Wrapper and Cognitive Experiment Utilities
#'
#' @description Provides functions to interact with multiple LLM APIs
#' (e.g., OpenAI, DeepSeek, Anthropic, Cohere, Google PaLM, Ollama).
#'
#' Additionally, several functions are provided that encapsulate LLM prompts to obtain
#' various psycholinguistic metrics:
#' \itemize{
#'   \item \strong{Word Frequency}: Number of occurrences (often log-transformed) of a word in a corpus.
#'   \item \strong{Lexical Coverage}: Proportion of words in the stimulus that are found in a target vocabulary.
#'   \item \strong{Zipf Metric}: The slope of the log-log frequency vs. rank distribution, per Zipf's law.
#'   \item \strong{Levenshtein Distance (D)}: Minimum number of single-character edits required to transform one stimulus into another.
#'   \item \strong{Semantic Transparency}: The degree to which the meaning of a compound or phrase is inferable from its parts.
#' }
#'
#' @details The LLM API functions allow users to generate ratings using various large language models.
#'
#' @import httr
#' @import jsonlite
#' @importFrom stats optim
#' @importFrom utils write.csv
#'
#' @name chatRater-package
NULL

# ------------------------------------------------------------------------------
# Base API Call Wrapper
# ------------------------------------------------------------------------------

#' @title Base LLM API Call Wrapper
#'
#' @description Sends a prompt (with background academic definitions) to an LLM API
#' (defaulting to OpenAI) and returns the LLM response.
#'
#' @param prompt_text A character string containing the user prompt.
#' @param model A character string specifying the LLM model (default "gpt-3.5-turbo").
#' @param api_key API key as a character string.
#' @param top_p Numeric value for the probability mass (default 1).
#' @param temp Numeric value for the sampling temperature (default 0).
#'
#' @return A character string containing the LLM's response.
#'
#' @details The system prompt includes academic definitions for word frequency, lexical coverage,
#' Zipf's law, Levenshtein distance, and semantic transparency.
#'
#' @examples
#' \dontrun{
#'   response <- llm_api_call("Please provide a rating for the stimulus 'apple'.")
#'   cat(response)
#' }
#' @export
llm_api_call <- function(prompt_text, model = "gpt-3.5-turbo", api_key = "", top_p = 1, temp = 0) {
  res <- openai::create_chat_completion(
    model = model,
    openai_api_key = api_key,
    top_p = top_p,
    temperature = temp,
    messages = list(
      list(role = "system",
           content = paste(
             "You are a knowledgeable evaluation assistant. Use the following academic definitions as your default background:",
             "Word frequency is defined as the number of times a word appears (often log-transformed).",
             "Lexical coverage is the proportion of words known in a text.",
             "Zipf's law states that word frequency is inversely proportional to its rank.",
             "Levenshtein distance is the minimum number of single-character edits needed to convert one string into another.",
             "Semantic transparency is the extent to which the meaning of a compound can be inferred from its parts."
           )),
      list(role = "user", content = prompt_text)
    )
  )
  # Return the content of the assistant's response.
  return(res$choices[1, "message.content"])
}

# ------------------------------------------------------------------------------
# Multi-Model API Functions for Rating Stimuli
# ------------------------------------------------------------------------------

#' @title Generate Ratings for a Stimulus Using LLM APIs
#'
#' @description Generates ratings for a given stimulus by calling one of several LLM APIs.
#'
#' @param model A character string specifying the LLM model (e.g., "gpt-3.5-turbo", "deepseek-chat").
#' @param stim A character string representing the stimulus (e.g., an idiom).
#' @param prompt A character string for the system prompt (e.g., "You are a native English speaker.").
#' @param question A character string for the user prompt (e.g., "Please rate the following stim:").
#' @param top_p Numeric value for the probability mass (default 1).
#' @param temp Numeric value for the temperature (default 0).
#' @param n_iterations An integer indicating how many times to query the API.
#' @param api_key API key as a character string.
#' @param debug Logical; if TRUE, debug information is printed.
#'
#' @return A data frame containing the stimulus, the rating, and the iteration number.
#'
#' @details This function supports multiple APIs. Branching is based on the \code{model} parameter.
#'
#' @examples
#' \dontrun{
#'   ratings <- generate_ratings(
#'     model = "gpt-3.5-turbo",
#'     stim = "kick the bucket",
#'     prompt = "You are a native English speaker.",
#'     question = "Please rate the following stim:",
#'     n_iterations = 5,
#'     api_key = "your_api_key"
#'   )
#'   print(ratings)
#' }
#' @export
generate_ratings <- function(model = 'gpt-3.5-turbo',
                             stim = 'kick the bucket',
                             prompt = 'You are a native English speaker.',
                             question = 'Please rate the following stim:',
                             top_p = 1,
                             temp = 0,
                             n_iterations = 30,
                             api_key = '',
                             debug = FALSE) {
  results <- data.frame(stim = character(), rating = integer(), iteration = integer(), stringsAsFactors = FALSE)
  Sys.setenv("OPENAI_API_KEY" = api_key)

  for (i in seq_len(n_iterations)) {
    combined_text <- paste("The stim is:", stim)

    if (model == "deepseek-chat") {
      url <- "https://api.deepseek.com/chat/completions"
      body <- list(
        model = model,
        top_p = top_p,
        temperature = temp,
        messages = list(
          list(role = "system", content = prompt),
          list(role = "user", content = question),
          list(role = "user", content = combined_text)
        )
      )
      res_raw <- httr::POST(
        url = url,
        add_headers(Authorization = paste("Bearer", api_key),
                    `Content-Type` = "application/json"),
        body = jsonlite::toJSON(body, auto_unbox = TRUE)
      )
      if (httr::status_code(res_raw) != 200) {
        warning("DeepSeek API call failed with status code: ", httr::status_code(res_raw))
        next
      }
      res <- httr::content(res_raw, as = "parsed", type = "application/json")
      rating_text <- res$choices[1, "message.content"]

    } else if (model == "anthropic") {
      url <- "https://api.anthropic.com/v1/complete"
      body <- list(
        prompt = paste("System:", prompt, "\nUser:", question, "\nStim:", combined_text),
        model = model,
        max_tokens_to_sample = 10,
        temperature = temp,
        top_p = top_p
      )
      res_raw <- httr::POST(
        url = url,
        add_headers(`X-API-Key` = api_key,
                    `Content-Type` = "application/json"),
        body = jsonlite::toJSON(body, auto_unbox = TRUE)
      )
      if (httr::status_code(res_raw) != 200) {
        warning("Anthropic API call failed with status code: ", httr::status_code(res_raw))
        next
      }
      res <- httr::content(res_raw, as = "parsed", type = "application/json")
      rating_text <- res$completion

    } else if (model == "cohere") {
      url <- "https://api.cohere.ai/generate"
      body <- list(
        model = model,
        prompt = paste("System:", prompt, "\nUser:", question, "\nStim:", combined_text),
        max_tokens = 5,
        temperature = temp,
        p = top_p
      )
      res_raw <- httr::POST(
        url = url,
        add_headers(Authorization = paste("Bearer", api_key),
                    `Content-Type` = "application/json"),
        body = jsonlite::toJSON(body, auto_unbox = TRUE)
      )
      if (httr::status_code(res_raw) != 200) {
        warning("Cohere API call failed with status code: ", httr::status_code(res_raw))
        next
      }
      res <- httr::content(res_raw, as = "parsed", type = "application/json")
      rating_text <- res$generations[[1]]$text

    } else if (model == "google-palm") {
      base_url <- "https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generate"
      url <- paste0(base_url, "?key=", api_key)
      body <- list(
        prompt = list(text = paste(prompt, question, combined_text)),
        temperature = temp,
        top_p = top_p,
        candidate_count = 1
      )
      res_raw <- httr::POST(
        url = url,
        add_headers(`Content-Type` = "application/json"),
        body = jsonlite::toJSON(body, auto_unbox = TRUE)
      )
      if (httr::status_code(res_raw) != 200) {
        warning("Google PaLM API call failed with status code: ", httr::status_code(res_raw))
        next
      }
      res <- httr::content(res_raw, as = "parsed", type = "application/json")
      rating_text <- res$candidates[[1]]$output

    } else if (model == "ollama") {
      url <- "http://localhost:11434/api/generate"
      body <- list(
        model = model,
        prompt = paste(prompt, question, combined_text),
        temperature = temp,
        top_p = top_p
      )
      res_raw <- httr::POST(
        url = url,
        add_headers(`Content-Type` = "application/json"),
        body = jsonlite::toJSON(body, auto_unbox = TRUE)
      )
      if (httr::status_code(res_raw) != 200) {
        warning("Ollama API call failed with status code: ", httr::status_code(res_raw))
        next
      }
      res <- httr::content(res_raw, as = "parsed", type = "application/json")
      rating_text <- res$output

    } else {
      res <- openai::create_chat_completion(
        model = model,
        openai_api_key = api_key,
        top_p = top_p,
        temperature = temp,
        messages = list(
          list(role = "system", content = prompt),
          list(role = "user", content = question),
          list(role = "user", content = combined_text)
        )
      )
      rating_text <- res$choices[1, "message.content"]
    }

    if (debug) {
      cat("Iteration:", i, "\n")
      cat("Stim:", stim, "\n")
      cat("Combined Text:", combined_text, "\n")
      cat("API Response:\n")
      utils::str(res)
    }

    rating <- as.integer(trimws(rating_text))
    if (is.na(rating)) {
      warning("Rating is NA for stim: ", stim, " at iteration: ", i)
    } else {
      results <- rbind(results, data.frame(stim = stim, rating = rating, iteration = i, stringsAsFactors = FALSE))
    }
  }
  return(results)
}

#' @title Generate Ratings for Multiple Stimuli
#'
#' @description Applies the \code{generate_ratings} function to a list of stimuli.
#'
#' @param model A character string specifying the LLM model.
#' @param stim_list A character vector of stimuli.
#' @param prompt A character string for the system prompt.
#' @param question A character string for the user prompt.
#' @param top_p Numeric value for the probability mass.
#' @param temp Numeric value for the temperature.
#' @param n_iterations An integer indicating how many iterations per stimulus.
#' @param api_key API key as a character string.
#' @param debug Logical; if TRUE, debug information is printed.
#'
#' @return A data frame with stimuli, ratings, and iteration numbers.
#'
#' @examples
#' \dontrun{
#'   all_ratings <- generate_ratings_for_all(
#'     model = "gpt-3.5-turbo",
#'     stim_list = c("kick the bucket", "spill the beans"),
#'     prompt = "You are a native English speaker.",
#'     question = "Please rate the following stim:",
#'     n_iterations = 5,
#'     api_key = "your_api_key"
#'   )
#'   print(all_ratings)
#' }
#' @export
generate_ratings_for_all <- function(model = 'gpt-3.5-turbo', stim_list,
                                     prompt = 'You are a native English speaker.',
                                     question = 'Please rate the following stim:',
                                     top_p = 1, temp = 0, n_iterations = 30,
                                     api_key = '', debug = FALSE) {
  all_results <- data.frame(stim = character(), rating = integer(), iteration = integer(), stringsAsFactors = FALSE)
  for (stim in stim_list) {
    results <- generate_ratings(model, stim, prompt, question, top_p, temp, n_iterations, api_key, debug)
    all_results <- rbind(all_results, results)
  }
  return(all_results)
}

# ------------------------------------------------------------------------------
# Psycholinguistic Metric Functions using LLM Prompts
# ------------------------------------------------------------------------------

#' @title Get Word Frequency Information
#'
#' @description Uses an LLM to obtain frequency information for a specified word position
#' in the stimulus. The user can specify a corpus; if none is provided and corpus_source is "llm",
#' the LLM will generate or assume a representative corpus.
#'
#' @param stimulus A character string representing the language material.
#' @param position A character string indicating which word to analyze ("first", "last", "each", or "total").
#' @param corpus An optional character string representing the corpus to use for frequency analysis.
#' @param corpus_source A character string, either "provided" or "llm". Default is "provided" if corpus is given, otherwise "llm".
#' @param model A character string specifying the LLM model (default "gpt-3.5-turbo").
#' @param api_key API key as a character string.
#' @param top_p Numeric value for probability mass (default 1).
#' @param temp Numeric value for temperature (default 0).
#'
#' @return A numeric value representing the frequency (or a JSON string if "each" is specified).
#'
#' @details Default definition: "Word frequency is defined as the number of times a word appears in a corpus (often log-transformed)."
#'
#' @examples
#' \dontrun{
#'   freq_first <- get_word_frequency("The quick brown fox jumps over the lazy dog",
#'                                    position = "first",
#'                                    corpus = "A sample corpus text with everyday language.",
#'                                    corpus_source = "provided",
#'                                    model = "gpt-3.5-turbo",
#'                                    api_key = "your_api_key")
#'   cat("Frequency (first word):", freq_first, "\n")
#' }
#' @export
get_word_frequency <- function(stimulus, position = "first",
                               corpus = "",
                               corpus_source = ifelse(corpus != "", "provided", "llm"),
                               model = "gpt-3.5-turbo", api_key = "",
                               top_p = 1, temp = 0) {
  corpus_instruction <- if (corpus_source == "provided" && corpus != "") {
    paste("Use the provided corpus for analysis. Corpus:", corpus)
  } else {
    "Please generate or assume a representative corpus of standard English usage for frequency analysis."
  }

  prompt_text <- paste(
    "Default knowledge: Word frequency is defined as the number of times a word appears in a corpus (often log-transformed).",
    corpus_instruction,
    "Analyze the following stimulus and provide frequency information for the word specified by the position parameter.",
    "Position options: 'first' (first word), 'last' (last word), 'each' (frequency for each word), or 'total' (total word count).",
    "\nStimulus:", stimulus,
    "\nPosition:", position
  )

  response_text <- llm_api_call(prompt_text, model = model, api_key = api_key, top_p = top_p, temp = temp)
  freq <- as.numeric(trimws(response_text))
  if (is.na(freq)) {
    warning("Word frequency could not be parsed. Response: ", response_text)
  }
  return(freq)
}

#' @title Get Lexical Coverage with Specified Vocabulary
#'
#' @description Uses an LLM to obtain the lexical coverage (percentage) of a given text,
#' taking into account a specified vocabulary size and the vocabulary test basis.
#'
#' @param stimulus A character string representing the language material.
#' @param vocab_size A numeric value indicating the size of the target vocabulary (e.g., 1000, 2000, 3000).
#' @param vocab_test A character string specifying the vocabulary test used (e.g., "Vocabulary Levels Test", "LexTALE").
#'        Users may provide any test name.
#' @param model A character string specifying the LLM model (default "gpt-3.5-turbo").
#' @param api_key API key as a character string.
#' @param top_p Numeric value for probability mass (default 1).
#' @param temp Numeric value for temperature (default 0).
#'
#' @return A numeric value indicating the lexical coverage percentage.
#'
#' @details Default definition: "Lexical coverage is the proportion of words in a text that are included in a given vocabulary list.
#' For this evaluation, assume a target vocabulary size of \code{vocab_size} words based on the \code{vocab_test}."
#'
#' @examples
#' \dontrun{
#'   coverage <- get_lexical_coverage("The quick brown fox jumps over the lazy dog",
#'                                    vocab_size = 2000,
#'                                    vocab_test = "Vocabulary Levels Test",
#'                                    model = "gpt-3.5-turbo",
#'                                    api_key = "your_api_key")
#'   cat("Lexical Coverage (%):", coverage, "\n")
#' }
#' @export
get_lexical_coverage <- function(stimulus, vocab_size = 2000, vocab_test = "Vocabulary Levels Test",
                                 model = "gpt-3.5-turbo", api_key = "", top_p = 1, temp = 0) {
  prompt_text <- paste(
    "Default knowledge: Lexical coverage is defined as the proportion of words in a text that are included in a target vocabulary.",
    "For this evaluation, assume a target vocabulary size of", vocab_size, "words based on the", vocab_test, ".",
    "Compute the lexical coverage (as a percentage) for the following stimulus.",
    "\nStimulus:", stimulus
  )

  response_text <- llm_api_call(prompt_text, model = model, api_key = api_key, top_p = top_p, temp = temp)
  coverage <- as.numeric(trimws(response_text))
  if (is.na(coverage)) {
    warning("Lexical coverage could not be parsed. Response: ", response_text)
  }
  return(coverage)
}

#' @title Get Zipf Metric
#'
#' @description Uses an LLM to estimate a Zipf-based metric (slope) for the given stimulus.
#'
#' @param stimulus A character string representing the language material.
#' @param model A character string specifying the LLM model (default "gpt-3.5-turbo").
#' @param api_key API key as a character string.
#' @param top_p Numeric value (default 1).
#' @param temp Numeric value (default 0).
#'
#' @return A numeric value representing the Zipf metric.
#'
#' @details Default definition: "Zipf's law states that word frequency is inversely proportional to its rank;
#' the Zipf metric is the slope of the log-log frequency vs. rank plot."
#'
#' @examples
#' \dontrun{
#'   zipf_metric <- get_zipf_metric("The quick brown fox jumps over the lazy dog",
#'                                  model = "gpt-3.5-turbo",
#'                                  api_key = "your_api_key")
#'   cat("Zipf Metric:", zipf_metric, "\n")
#' }
#' @export
get_zipf_metric <- function(stimulus, model = "gpt-3.5-turbo",
                            api_key = "", top_p = 1, temp = 0) {
  prompt_text <- paste(
    "Default knowledge: Zipf's law states that the frequency of a word is inversely proportional to its rank. ",
    "The Zipf metric is the slope of the log-log plot of frequency against rank.",
    "Estimate the Zipf metric for the following stimulus.",
    "\nStimulus:", stimulus
  )
  response_text <- llm_api_call(prompt_text, model = model, api_key = api_key, top_p = top_p, temp = temp)
  zipf_value <- as.numeric(trimws(response_text))
  if (is.na(zipf_value)) {
    warning("Zipf metric could not be parsed. Response: ", response_text)
  }
  return(zipf_value)
}

#' @title Get Levenshtein Distance (D)
#'
#' @description Uses an LLM to compute the Levenshtein distance (D) between two linguistic stimuli.
#'
#' @param stimulus1 A character string representing the first text.
#' @param stimulus2 A character string representing the second text.
#' @param model A character string specifying the LLM model (default "gpt-3.5-turbo").
#' @param api_key API key as a character string.
#' @param top_p Numeric value (default 1).
#' @param temp Numeric value (default 0).
#'
#' @return A numeric value representing the Levenshtein distance.
#'
#' @details Default definition: "Levenshtein distance is defined as the minimum number of single-character edits
#' (insertions, deletions, or substitutions) required to transform one string into another."
#'
#' @examples
#' \dontrun{
#'   lev_dist <- get_levenshtein_d("kitten", "sitting",
#'                                 model = "gpt-3.5-turbo",
#'                                 api_key = "your_api_key")
#'   cat("Levenshtein Distance:", lev_dist, "\n")
#' }
#' @export
get_levenshtein_d <- function(stimulus1, stimulus2, model = "gpt-3.5-turbo",
                              api_key = "", top_p = 1, temp = 0) {
  prompt_text <- paste(
    "Default knowledge: Levenshtein distance is defined as the minimum number of single-character edits (insertions, deletions, substitutions) needed to change one string into another.",
    "Compute the Levenshtein D between the following two stimuli.",
    "\nStimulus 1:", stimulus1,
    "\nStimulus 2:", stimulus2,
    "\nReturn a single numeric value."
  )
  response_text <- llm_api_call(prompt_text, model = model, api_key = api_key, top_p = top_p, temp = temp)
  lev_d <- as.numeric(trimws(response_text))
  if (is.na(lev_d)) {
    warning("Levenshtein D could not be parsed. Response: ", response_text)
  }
  return(lev_d)
}

#' @title Get Semantic Transparency Rating
#'
#' @description Uses an LLM to obtain a semantic transparency rating for the given linguistic stimulus.
#'
#' @param stimulus A character string representing the language material.
#' @param model A character string specifying the LLM model (default "gpt-3.5-turbo").
#' @param api_key API key as a character string.
#' @param top_p Numeric value (default 1).
#' @param temp Numeric value (default 0).
#'
#' @return An integer rating (1-7) indicating the semantic transparency.
#'
#' @details Default definition: "Semantic transparency is the degree to which the meaning of a compound or phrase
#' can be inferred from its constituent parts."
#'
#' @examples
#' \dontrun{
#'   sem_trans <- get_semantic_transparency("blackbird",
#'                                           model = "gpt-3.5-turbo",
#'                                           api_key = "your_api_key")
#'   cat("Semantic Transparency Rating:", sem_trans, "\n")
#' }
#' @export
get_semantic_transparency <- function(stimulus, model = "gpt-3.5-turbo",
                                      api_key = "", top_p = 1, temp = 0) {
  prompt_text <- paste(
    "Default knowledge: Semantic transparency refers to how readily the meaning of a compound or phrase can be derived from its parts.",
    "Provide a rating on a scale of 1 (opaque) to 7 (fully transparent) for the following stimulus.",
    "\nStimulus:", stimulus
  )
  response_text <- llm_api_call(prompt_text, model = model, api_key = api_key, top_p = top_p, temp = temp)
  transparency <- as.integer(trimws(response_text))
  if (is.na(transparency)) {
    warning("Semantic transparency rating could not be parsed. Response: ", response_text)
  }
  return(transparency)
}

# ------------------------------------------------------------------------------
# End of Package Code
# ------------------------------------------------------------------------------

## Examples of Usage (for package vignettes or manual testing):
## Uncomment and replace "your_api_key" with a valid API key to test these functions.

#' @examples
#' \dontrun{
#'   # Generate ratings for a stimulus
#'   ratings <- generate_ratings(
#'     model = "gpt-3.5-turbo",
#'     stim = "kick the bucket",
#'     prompt = "You are a native English speaker.",
#'     question = "Please rate the following stim:",
#'     n_iterations = 5,
#'     api_key = "your_api_key"
#'   )
#'   print(ratings)
#'
#'   # Generate ratings for multiple stimuli
#'   all_ratings <- generate_ratings_for_all(
#'     model = "gpt-3.5-turbo",
#'     stim_list = c("kick the bucket", "spill the beans"),
#'     prompt = "You are a native English speaker.",
#'     question = "Please rate the following stim:",
#'     n_iterations = 5,
#'     api_key = "your_api_key"
#'   )
#'   print(all_ratings)
#'
#'   # Get word frequency (first word) with a provided corpus
#'   freq_first <- get_word_frequency(
#'     stimulus = "The quick brown fox jumps over the lazy dog",
#'     position = "first",
#'     corpus = "A sample corpus text with everyday language.",
#'     corpus_source = "provided",
#'     model = "gpt-3.5-turbo",
#'     api_key = "your_api_key"
#'   )
#'   cat("Frequency (first word):", freq_first, "\n")
#'
#'   # Get lexical coverage based on a target vocabulary size and test
#'   coverage <- get_lexical_coverage(
#'     stimulus = "The quick brown fox jumps over the lazy dog",
#'     vocab_size = 2000,
#'     vocab_test = "Vocabulary Levels Test",
#'     model = "gpt-3.5-turbo",
#'     api_key = "your_api_key"
#'   )
#'   cat("Lexical Coverage (%):", coverage, "\n")
#'
#'   # Get Zipf metric for a stimulus
#'   zipf_metric <- get_zipf_metric(
#'     stimulus = "The quick brown fox jumps over the lazy dog",
#'     model = "gpt-3.5-turbo",
#'     api_key = "your_api_key"
#'   )
#'   cat("Zipf Metric:", zipf_metric, "\n")
#'
#'   # Compute Levenshtein distance between two strings
#'   lev_dist <- get_levenshtein_d(
#'     stimulus1 = "kitten",
#'     stimulus2 = "sitting",
#'     model = "gpt-3.5-turbo",
#'     api_key = "your_api_key"
#'   )
#'   cat("Levenshtein Distance:", lev_dist, "\n")
#'
#'   # Get semantic transparency rating for a stimulus
#'   sem_trans <- get_semantic_transparency(
#'     stimulus = "blackbird",
#'     model = "gpt-3.5-turbo",
#'     api_key = "your_api_key"
#'   )
#'   cat("Semantic Transparency Rating:", sem_trans, "\n")
#' }
