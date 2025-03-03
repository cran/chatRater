% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/chatRater.R
\name{get_zipf_metric}
\alias{get_zipf_metric}
\title{Get Zipf Metric}
\usage{
get_zipf_metric(
  stimulus,
  model = "gpt-3.5-turbo",
  api_key = "",
  top_p = 1,
  temp = 0
)
}
\arguments{
\item{stimulus}{A character string representing the language material.}

\item{model}{A character string specifying the LLM model (default "gpt-3.5-turbo").}

\item{api_key}{API key as a character string.}

\item{top_p}{Numeric value (default 1).}

\item{temp}{Numeric value (default 0).}
}
\value{
A numeric value representing the Zipf metric.
}
\description{
Uses an LLM to estimate a Zipf-based metric (slope) for the given stimulus.
}
\details{
Default definition: "Zipf's law states that word frequency is inversely proportional to its rank;
the Zipf metric is the slope of the log-log frequency vs. rank plot."
}
\examples{
\dontrun{
  zipf_metric <- get_zipf_metric("The quick brown fox jumps over the lazy dog",
                                 model = "gpt-3.5-turbo",
                                 api_key = "your_api_key")
  cat("Zipf Metric:", zipf_metric, "\n")
}
}
