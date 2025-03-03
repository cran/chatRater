% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/chatRater.R
\name{generate_ratings_for_all}
\alias{generate_ratings_for_all}
\title{Generate Ratings for Multiple Stimuli}
\usage{
generate_ratings_for_all(
  model = "gpt-3.5-turbo",
  stim_list,
  prompt = "You are a native English speaker.",
  question = "Please rate the following stim:",
  top_p = 1,
  temp = 0,
  n_iterations = 30,
  api_key = "",
  debug = FALSE
)
}
\arguments{
\item{model}{A character string specifying the LLM model.}

\item{stim_list}{A character vector of stimuli.}

\item{prompt}{A character string for the system prompt.}

\item{question}{A character string for the user prompt.}

\item{top_p}{Numeric value for the probability mass.}

\item{temp}{Numeric value for the temperature.}

\item{n_iterations}{An integer indicating how many iterations per stimulus.}

\item{api_key}{API key as a character string.}

\item{debug}{Logical; if TRUE, debug information is printed.}
}
\value{
A data frame with stimuli, ratings, and iteration numbers.
}
\description{
Applies the \code{generate_ratings} function to a list of stimuli.
}
\examples{
\dontrun{
  all_ratings <- generate_ratings_for_all(
    model = "gpt-3.5-turbo",
    stim_list = c("kick the bucket", "spill the beans"),
    prompt = "You are a native English speaker.",
    question = "Please rate the following stim:",
    n_iterations = 5,
    api_key = "your_api_key"
  )
  print(all_ratings)
}
}
