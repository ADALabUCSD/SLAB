BENCHMARK_PROJECT_ROOT <- Sys.getenv('BENCHMARK_PROJECT_ROOT')
source(paste(BENCHMARK_PROJECT_ROOT, '/lib/R/R_timing_utils.R', sep = ''))

doMatrixOp <- function(argList) {
    colnames <- c('rows', paste('time',1:5, sep=''))
    runTimes <- as.data.frame(matrix(0, nrow = 1, ncol = length(colnames)))
    names(runTimes) <- colnames
    
    if (!is.numeric(argList[['nrows']]))
        nrows <- sapply(
                   X=unlist(strsplit(argList[['nrows']], ' ')), FUN=as.numeric)
    else
        nrows <- argList[['nrows']]
    opType <- argList[['opType']]
    mattype <- argList[['mattype']]
    fixedAxis <- argList[['fixedAxis']]
    nproc <- argList[['nproc']]

    if (opType == 'TRANS') {
        call <- 't(M)'
    } else if (opType == 'NORM') {
        call <- 'norm(M, type="F")'
    } else if (opType == 'GMM') {
        call <- 'M %*% N'
    } else if (opType == 'MVM') {
        call <- 'M %*% w'
    } else if (opType == 'TSM') {
        call <- 'crossprod(M)'
    } else if (opType == 'ADD') {
        call <- 'M + N'
    } else {
        cat('Invalid operation\n')
        stop()
    }

    if (is.null(nproc)) {
        path <- paste('../output/R_', mattype, '_', opType, '.txt', sep='')
    } else {
        path <- paste('../output/R_cpu', opType, '_scale.txt', sep='')
    }
    for (nr in nrows) {
        nrow <- if (opType == 'GMM') fixedAxis else nr
        ncol <- if (opType == 'GMM') nr else fixedAxis
        majorAxis <- if (opType == 'GMM') ncol else nrow

        M <- allocMatrix(nrow, ncol)

        if (opType == 'GMM') {
            N <- allocMatrix(ncol, nrow)
        } else if (opType == 'MVM') {
            w <- allocMatrix(ncol, 1)
        } else if (opType == 'ADD') {
            N <- allocMatrix(nrow, ncol)
        }

        runTimes[1,'rows'] <- nr
        runTimes[1,paste('time', 1:5, sep='')] <- timeOp(call)
        writeHeader <- if (!file.exists(path)) TRUE else FALSE
        write.table(runTimes, 
                    path, 
                    append = TRUE, 
                    row.names = FALSE,
                    col.names = writeHeader,
                    sep = ',')
    }

}

argv <- commandArgs(trailingOnly = FALSE)
args <- list()
for (arg in argv[1:length(argv)]) {
    parsed <- parseCMDArg(arg)
    args[[parsed[[1]]]] <- parsed[[2]]
}

doMatrixOp(args)
