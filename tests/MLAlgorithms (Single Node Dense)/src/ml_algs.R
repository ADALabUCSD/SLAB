BENCHMARK_PROJECT_ROOT <- Sys.getenv('BENCHMARK_PROJECT_ROOT')
source(paste(BENCHMARK_PROJECT_ROOT, '/lib/R/R_timing_utils.R', sep = ''))

main <- function(argv) {
    mattype <- argList[['mattype']]
    opType <- argList[['opType']]
    nrow <- as.numeric(argList[['nrow']])
    ncol <- as.numeric(argList[['ncol']])
    nproc <- as.numeric(argList[['nproc']])
    path <- paste('../output/R_', opType, '.txt', sep='')

    colnames <- c('nproc','time1','time2','time3','time4','time5')
    runTimes <- as.data.frame(matrix(0, nrow = 1, ncol = length(colnames)))
    names(runTimes) <- colnames

    X <- allocMatrix(nrow, ncol)
    print(dim(X))
    if (opType != 'gnmf') {
        y <- allocMatrix(nrow, 1, TRUE)
    }
    if (opType == 'logit') {
        call <- 'logitReg(X,y)'
    } else if (opType == 'reg') {
        call <- 'reg(X,y)'
    } else if (opType == 'gnmf') {
        call <- 'gnmf(X,10)'
    } else if (opType == 'robust') {
        b <- reg(X,y)
        y_hat <- X %*% b
        eps <- as.vector(y_hat^2)
        print(length(eps))
        call <- 'robust_se(X,eps)'
    }

    runTimes[1,'nproc'] <- nproc
    runTimes[1,2:ncol(runTimes)] <- timeOp(call)
    writeHeader <- if (!file.exists(path)) TRUE else FALSE
    write.table(runTimes,
                path,
                append = TRUE,
                row.names = FALSE,
                col.names = writeHeader,
                sep = ',')
}

allocMatrix <- function(rows, cols, binary=FALSE) {
    if (binary) {
        M <- as.numeric(matrix(runif(rows*cols), nrow=rows, ncol=cols) >= .80)
    } else {
        M <- matrix(rnorm(rows*cols), nrow=rows, ncol=cols)
    }
    return(M)
}

logitReg <- function(X, y, iterations=3) {
    N <- nrow(X)
    w <- allocMatrix(ncol(X),1)
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
    W <- allocMatrix(nrow(X), r)
    H <- allocMatrix(r, ncol(X))

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
    S <- sweep(t(X), MARGIN=2, STATS=r2, FUN='*')
    XTX_INV <- solve( crossprod( X ) )
    se <- XTX_INV %*% (S %*% X) %*% XTX_INV
    return(se)
}

argv <- commandArgs(trailingOnly = TRUE)
argList <- list()
for (arg in argv) {
    parsed <- suppressMessages(parseCMDArg(arg))
    argList[[parsed[[1]]]] <- parsed[[2]]
}

main()
