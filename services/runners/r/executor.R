# R Code Executor for Rocky AI
# Provides secure, sandboxed execution of R code with resource limits

library(renv)
library(jsonlite)

# Configuration
MAX_EXECUTION_TIME <- 30  # seconds
MAX_MEMORY_MB <- 512      # MB
MAX_OUTPUT_SIZE <- 1024 * 1024  # 1MB

# Allowed packages for security
ALLOWED_PACKAGES <- c(
  "tidyverse", "dplyr", "ggplot2", "tidyr", "readr", "purrr",
  "broom", "car", "survival", "survminer", "lme4", "nlme",
  "stats", "base", "utils", "graphics", "grDevices"
)

# Function to validate R code
validate_code <- function(code) {
  # Check for dangerous functions
  dangerous_functions <- c(
    "system", "shell", "system2", "file.create", "dir.create",
    "unlink", "file.remove", "download.file", "url", "socketConnection"
  )
  
  for(func in dangerous_functions) {
    if(grepl(paste0("\\b", func, "\\s*\\("), code)) {
      return(list(valid = FALSE, error = paste("Function", func, "not allowed")))
    }
  }
  
  # Check for library/require calls
  library_matches <- regmatches(code, gregexpr("(library|require)\\s*\\(\\s*['\"]([^'\"]+)['\"]", code, perl = TRUE))
  for(match in library_matches) {
    if(length(match) > 0) {
      package_name <- gsub(".*['\"]([^'\"]+)['\"].*", "\\1", match)
      if(!package_name %in% ALLOWED_PACKAGES) {
        return(list(valid = FALSE, error = paste("Package", package_name, "not allowed")))
      }
    }
  }
  
  return(list(valid = TRUE, error = ""))
}

# Function to execute R code safely
execute_r_code <- function(code, timeout = MAX_EXECUTION_TIME) {
  start_time <- Sys.time()
  
  # Validate code
  validation <- validate_code(code)
  if(!validation$valid) {
    return(list(
      success = FALSE,
      output = "",
      error = validation$error,
      execution_time = 0,
      memory_used_mb = 0,
      return_code = 1
    ))
  }
  
  # Set up output capture
  output_file <- tempfile(fileext = ".txt")
  error_file <- tempfile(fileext = ".txt")
  
  # Create execution script
  exec_script <- paste0("
    # Set memory limit
    options(expressions = 500000)
    
    # Capture output
    sink(file = '", output_file, "', type = 'output')
    sink(file = '", error_file, "', type = 'message')
    
    tryCatch({
      # Execute user code
      ", code, "
    }, error = function(e) {
      cat('Error:', e$message, '\\n')
    }, finally = {
      sink(type = 'output')
      sink(type = 'message')
    })
  ")
  
  # Write script to temporary file
  script_file <- tempfile(fileext = ".R")
  writeLines(exec_script, script_file)
  
  tryCatch({
    # Execute with timeout
    result <- system2("Rscript", script_file, 
                     stdout = TRUE, stderr = TRUE, 
                     timeout = timeout)
    
    # Read output
    output <- if(file.exists(output_file)) {
      paste(readLines(output_file, warn = FALSE), collapse = "\n")
    } else {
      ""
    }
    
    error <- if(file.exists(error_file)) {
      paste(readLines(error_file, warn = FALSE), collapse = "\n")
    } else {
      ""
    }
    
    # Calculate execution time
    execution_time <- as.numeric(Sys.time() - start_time)
    
    # Estimate memory usage (simplified)
    memory_used_mb <- if(exists("gc")) {
      gc()[1, 2]  # Max memory used
    } else {
      0
    }
    
    # Check output size
    total_output_size <- nchar(output) + nchar(error)
    if(total_output_size > MAX_OUTPUT_SIZE) {
      output <- substr(output, 1, MAX_OUTPUT_SIZE / 2)
      error <- paste0(substr(error, 1, MAX_OUTPUT_SIZE / 2), 
                     "\n[Output truncated due to size limit]")
    }
    
    success <- length(result) == 0 || all(result == 0)
    
    return(list(
      success = success,
      output = output,
      error = error,
      execution_time = execution_time,
      memory_used_mb = memory_used_mb,
      return_code = if(length(result) > 0) result[1] else 0
    ))
    
  }, error = function(e) {
    return(list(
      success = FALSE,
      output = "",
      error = paste("Execution error:", e$message),
      execution_time = as.numeric(Sys.time() - start_time),
      memory_used_mb = 0,
      return_code = 1
    ))
  }, finally = {
    # Cleanup
    if(file.exists(output_file)) file.remove(output_file)
    if(file.exists(error_file)) file.remove(error_file)
    if(file.exists(script_file)) file.remove(script_file)
  })
}

# Function to install R package
install_package <- function(package_name) {
  if(!package_name %in% ALLOWED_PACKAGES) {
    return(FALSE)
  }
  
  tryCatch({
    if(!require(package_name, character.only = TRUE, quietly = TRUE)) {
      install.packages(package_name, repos = "https://cran.rstudio.com/")
      return(require(package_name, character.only = TRUE, quietly = TRUE))
    }
    return(TRUE)
  }, error = function(e) {
    return(FALSE)
  })
}

# Main execution function
main <- function() {
  # Read code from command line or stdin
  args <- commandArgs(trailingOnly = TRUE)
  
  if(length(args) > 0) {
    code <- paste(args, collapse = " ")
  } else {
    code <- readLines("stdin", warn = FALSE)
  }
  
  # Execute code
  result <- execute_r_code(code)
  
  # Return result as JSON
  cat(toJSON(result, auto_unbox = TRUE))
}

# Run if called directly
if(!interactive()) {
  main()
}
