suppressMessages(library(pbdDMAT, quietly=TRUE))
suppressMessages(library(pbdMPI, quietly=TRUE))
BENCHMARK_PROJECT_ROOT <- Sys.getenv('BENCHMARK_PROJECT_ROOT')
source(paste(BENCHMARK_PROJECT_ROOT, '/lib/R/R_timing_utils.R', sep = ''))
source(paste(BENCHMARK_PROJECT_ROOT, '/lib/R/readDMM.R', sep=''))

init.grid()

argv <- commandArgs(trailingOnly = TRUE)
argList <- list()
for (arg in argv) {
    parsed <- suppressMessages(parseCMDArg(arg))
    argList[[parsed[[1]]]] <- parsed[[2]]
}

mattype <- argList[['mattype']]
opType <- argList[['opType']]
Xpath <- argList[['Xpath']]
Ypath <- argList[['Ypath']]
nodes <- argList[['nodes']]
dataPath <- argList[['dataPath']]

alloc_matrix <- function(path, bldim=32) {
    meta <- parseMetadata(path)
    M <- ddmatrix('rnorm', meta[['rows']], meta[['cols']], bldim=bldim)
    return(M)
}

alloc_binary_matrix <- function(path, bldim=32) {
    meta <- parseMetadata(path)
    M <- ddmatrix('runif', meta[['rows']], meta[['cols']], bldim=bldim)
    return(M > .83)
}

logitReg <- function(X, y, iterations=3) {
    N <- nrow(X)

    w <- ddmatrix('rnorm', nrow=ncol(X), ncol=1, bldim=bldim(X))
    iteration <- 1
    stepSize <- 10

    while (iteration < iterations) {
        xb <- X %*% w
        delta <- y - 1/(1+exp(-xb))
        stepSize <- stepSize / 2
        w <- w + ((stepSize*crossprod(X, delta))/N)

        iteration <- iteration+1
    }

    return(w)
}

gnmf <- function(X, r, iterations=3) {
    W <- ddmatrix('rnorm', nrow=nrow(X), ncol=r, bldim=bldim(X))
    H <- ddmatrix('rnorm', nrow=r, ncol=ncol(X), bldim=bldim(X))

    iteration <- 0
    while (iteration < iterations) {
        W <- W * ((X %*% t(H)) / (W %*% tcrossprod(H,H)))
        H <- H * ((t(W) %*% X) / (crossprod(W,W) %*% H))
        iteration <- iteration + 1
    }

    return(list(W,H))
}

reg <- function(X, y) {
    b <- solve(t(X) %*% X, t(X) %*% y)
    return(b)
}

robust_se <- function(X, r2) {
    S <- sweep(t( X ), MARGIN=1, STATS=as.vector(r2), FUN='*')
    XTX_INV <- solve( crossprod( X ) )
    se <- XTX_INV %*% (S %*% X) %*% XTX_INV
    return(se)
}

pca <- function(X) {
    N <- nrow( X )
    XS <- sweep(X, 2, colMeans(X))
    S <- (1/(N-1)) * (crossprod( XS ))
    eigs <- eigen( S )

    # NOTE: IMPORTANT ISSUE: The eigenvectors need to be sorted by their values
    # ScaLAPACK is -not- guaranteed to do this internally. It's not clear how to
    # do this though using the tools provided by pbdR! This doesn't really matter
    # for performance but is important for accuracy.

    PRJ <- XS %*% eigs$values[,1:k]
    return(PRJ)
}

X <- alloc_matrix(Xpath)
y <- alloc_binary_matrix(Ypath)

rows <- nrow(X)
cols <- ncol(X)

# unfortunately we need to implement timing from scratch here
# as it's not clear how MPI plays with R's environments

times <- rep(0,5)

b <- NULL
if (opType == 'robust') {
    b <- reg(X, y)
    y_hat <- X %*% b
    s <- sum(y_hat)
    r2 <- t(y - y_hat)^2
}

for (ix in 1:5) {
    comm.print("START")
    barrier()
    a <- Sys.time()

    if (opType == 'logit') {
        logitReg(X, y)
    } else if (opType == 'gnmf') {
        gnmf(X, 10)
    } else if (opType == 'reg') {
        reg(X, y)
    } else if (opType == 'robust') {
        robust_se(X, r2)
    }

    comm.print('AT BARRIER')
    barrier()
    b <- Sys.time()
    times[ix] <- as.numeric(b-a, units="secs")
    comm.print('STOP')
}

if (comm.rank() == 0) {
    path <- paste('../output/R_', mattype, '_', opType, nodes, '.txt', sep='')

    colnames <- c('nodes','rows','cols','time1','time2','time3','time4','time5')
    runTimes <- as.data.frame(matrix(0, nrow = 1, ncol = length(colnames)))
    names(runTimes) <- colnames

    runTimes[1,'nodes'] <- nodes
    runTimes[1,c('rows','cols')] <- c(rows,cols)
    runTimes[1,4:ncol(runTimes)] <- times
    writeHeader <- if (!file.exists(path)) TRUE else FALSE
    write.table(runTimes,
                path,
                append = TRUE,
                row.names = FALSE,
                col.names = writeHeader,
                sep = ',')
}

finalize()
