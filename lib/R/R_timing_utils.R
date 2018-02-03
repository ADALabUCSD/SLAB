timeOp <- function(callable) {
    times <- c()
    environment <- parent.frame(1)
    for (i in 1:5) {
        t <- system.time(eval(parse(text=callable), env=environment))
        times <- c(times, t['elapsed'])
    }
    if (any(is.na(times))) {
        return(NA)
    } else {
        return(times)
    }
}

allocMatrix <- function(rows, cols) {
    rows <- as.integer(rows)
    cols <- as.integer(cols)
    mat <- tryCatch(
            {
                return(matrix(runif(rows*cols), nrow=rows, ncol=cols))
            },
            error=function(err) {
                message(paste("Error allocating matrix:", err))
                return(NULL)
            }
        )
    return(mat)
}

parseCMDArg <- function(argStr) {
    arg <- strsplit(argStr, '=')
    value <- arg[[1]][2]
    second <- if (is.na(suppressWarnings(as.numeric(value)))) value else as.numeric(value)
    L <- list(arg[[1]][1], second)
    return(L)
}