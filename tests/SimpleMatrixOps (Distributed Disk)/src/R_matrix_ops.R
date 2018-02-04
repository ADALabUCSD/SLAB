suppressMessages(library(pbdDMAT))
suppressMessages(library(pbdMPI))
BENCHMARK_PROJECT_ROOT <- Sys.getenv('BENCHMARK_PROJECT_ROOT')
source(paste(BENCHMARK_PROJECT_ROOT, '/lib/R/R_timing_utils.R', sep = ''))
source(paste(BENCHMARK_PROJECT_ROOT, '/lib/R/readDMM.R', sep=''))

init.grid()

argv <- commandArgs(trailingOnly = TRUE)
argList <- list()
for (arg in argv) {
    parsed <- parseCMDArg(arg)
    argList[[parsed[[1]]]] <- parsed[[2]]
}

mattype <- argList[['mattype']]
opType <- argList[['opType']]
Mpath <- argList[['Mpath']]
Npath <- argList[['Npath']]
wPath <- argList[['wPath']]
nodes <- argList[['nodes']]
outdir <- argList[['outdir']]

# reading in giant CSV files in R is a terrible process so just
# allocate data on the fly in distributed memory
# this amounts to the same thing for testing purposes
alloc_matrix <- function(path, bldim=32) {
    meta <- parseMetadata(path)
    M <- ddmatrix('rnorm', meta[['rows']], meta[['cols']], bldim=bldim)
    return(M)
}

M <- alloc_matrix(Mpath)
if (opType == 'TRANS') {
    call <- 't(M)'
} else if (opType == 'NORM') {
    call <- 'norm(M, type="F")'
} else if (opType == 'GMM') {
    N <- alloc_matrix(Npath)
    call <- 'M %*% N'
} else if (opType == 'MVM') {
    w <- ddmatrix('rnorm', nrow=ncol(M), ncol=1, bldim=bldim(M))
    call <- 'M %*% w'
} else if (opType == 'TSM') {
    call <- 'crossprod(M)'
} else if (opType == 'ADD') {
    N <- alloc_matrix(Npath)
    call <- 'M + N'
} else {
    comm.print('Invalid operation')
    comm.stop()
}

# unfortunately we need to implement timing from scratch here
# as it's not clear how MPI plays with R's environments
times <- double(5)

for (ix in 1:5) {
    comm.print('SYNCHRONIZING')
    barrier()
    comm.print('START')
    a <- Sys.time()

    res <- eval(parse(text=call))
    
    barrier()
    comm.print('COMPLETE')
    b <- Sys.time()
    times[ix] <- as.numeric(b - a, units='secs')
}

if (comm.rank() == 0) {
    rows <- nrow(M)
    cols <- ncol(M)
    path <- paste('../output/', outdir,
                  '/R_', mattype, '_',
                  opType, nodes,
                  '.txt', sep='')

    colnames <- c('nodes', 'rows', 'cols', paste('time',1:5,sep=''))
    runTimes <- as.data.frame(matrix(0, nrow = 1, ncol = length(colnames)))
    names(runTimes) <- colnames

    runTimes[1,c('nodes','rows','cols')] <- c(nodes, rows, cols)
    runTimes[1, 4:ncol(runTimes)] <- times
    writeHeader <- if (!file.exists(path)) TRUE else FALSE
    write.table(runTimes,
                path,
                append = TRUE,
                row.names = FALSE,
                col.names = writeHeader,
                sep = ',')
}

finalize()
