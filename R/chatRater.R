#' Generate ratings for a single stim using LLM
#'
#' This function generates ratings for a given stimulus using a Large Language Model (LLM).
#' It supports both OpenAI and DeepSeek APIs. When the model parameter is set to "deepseek-chat",
#' the DeepSeek API endpoint will be used.
#'
#' @param model A character string specifying the LLM model to use. Use "deepseek-chat" to call the DeepSeek API.
#' @param stim A character string representing the stim (e.g., an idiom).
#' @param prompt A character string providing context or an identity for LLM (e.g., "You are a native English speaker.").
#' @param question A character string that provides instructions for LLM.
#' @param top_p A numeric value limiting token selection to a probability mass.
#' @param temp A numeric value specifying the temperature for the API call.
#' @param n_iterations An integer indicating the number of times to query LLM for the stim.
#' @param api_key Your OpenAI API key.
#' @param debug Logical, whether to run in debug mode. Defaults to FALSE.
#' @importFrom utils str
#' @import httr
#' @import jsonlite
#' @return A data frame containing the stim, rating, and iteration number for each API call.
#' @export
#'
#' @examples
#' \dontrun{
#'   generate_ratings(model = "gpt-3.5-turbo",
#'                    stim = "kick the bucket",
#'                    prompt = "You are a native English speaker.",
#'                    question = "Please rate the following stim:",
#'                    top_p = 1,
#'                    temp = 0,
#'                    n_iterations = 30,
#'                    api_key = "your_api_key",
#'                    debug = TRUE)
#' }
generate_ratings <- function(model = 'gpt-3.5-turbo',
                             stim = 'kick the bucket',
                             prompt = '...',
                             question = '...',
                             top_p = 1,
                             temp = 0,
                             n_iterations = 30,
                             api_key = '',
                             debug = FALSE) {
  Sys.setenv("OPENAI_API_KEY" = api_key)
  results <- data.frame(stim = character(), rating = integer(), iteration = integer(), stringsAsFactors = FALSE)

  for (i in seq_len(n_iterations)) {
    combined_text <- paste("The stim is:", stim)

    if(model == "deepseek-chat") {
      # DeepSeek API call
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
      if(httr::status_code(res_raw) != 200) {
        warning("DeepSeek API call failed with status code: ", httr::status_code(res_raw))
        next
      }
      res <- httr::content(res_raw, as = "parsed", type = "application/json")
    } else {
      # OpenAI API call
      res <- openai::create_chat_completion(
        model = model,
        openai_api_key = Sys.getenv("OPENAI_API_KEY"),
        top_p = top_p,
        temperature = temp,
        messages = list(
          list(role = "system", content = prompt),
          list(role = "user", content = question),
          list(role = "user", content = combined_text)
        )
      )
    }

    if(debug) {
      cat("Iteration:", i, "\n")
      cat("Stim:", stim, "\n")
      cat("Combined Text:", combined_text, "\n")
      cat("API Response:\n")
      utils::str(res)
    }

    # Extract rating from API response (assumes rating is in message.content)
    rating_content <- res$choices[[1]]$message$content
    rating <- as.integer(rating_content)

    if (is.na(rating)) {
      warning("Rating is NA for stim: ", stim, " at iteration: ", i)
    } else {
      results <- rbind(results, data.frame(stim = stim, rating = rating, iteration = i, stringsAsFactors = FALSE))
    }
  }
  return(results)
}

#' Generate ratings for all stims using LLM
#'
#' This function iterates over a vector of stims (e.g., idioms) and generates ratings for each
#' by calling the \code{generate_ratings} function. It aggregates all results into a single data frame.
#'
#' @param model A character string specifying the LLM model to use.
#' @param stim_list A character vector of stims (e.g., idioms) for which ratings will be generated.
#' @param prompt A character string providing context or an identity for LLM (e.g., "You are a native English speaker.").
#' @param question A character string that provides instructions for LLM.
#' @param top_p A numeric value limiting token selection to a probability mass.
#' @param temp A numeric value specifying the temperature for the API call.
#' @param n_iterations An integer indicating the number of times to query LLM for each stim.
#' @param api_key Your OpenAI API key.
#' @param debug Logical, whether to run in debug mode. Defaults to FALSE.
#' @return A data frame containing the stim, rating, and iteration number for each API call.
#' @importFrom utils str
#' @import httr
#' @import jsonlite
#' @export
#'
#' @examples
#' \dontrun{
#'   generate_ratings_for_all(model = "gpt-3.5-turbo",
#'                            stim_list = c("kick the bucket", "spill the beans"),
#'                            prompt = "You are a native English speaker.",
#'                            question = "Please rate the following stim:",
#'                            top_p = 1,
#'                            temp = 0,
#'                            n_iterations = 30,
#'                            api_key = "your_api_key",
#'                            debug = TRUE)
#' }
generate_ratings_for_all <- function(model = 'gpt-3.5-turbo', stim_list, prompt = '...', question = '...', top_p = 1, temp = 0, n_iterations = 30, api_key = '', debug = FALSE) {
  all_results <- data.frame(stim = character(), rating = integer(), iteration = integer(), stringsAsFactors = FALSE)

  for (stim in stim_list) {
    results <- generate_ratings(model, stim, prompt, question, top_p, temp, n_iterations, api_key, debug)
    all_results <- rbind(all_results, results)
  }

  return(all_results)
}
