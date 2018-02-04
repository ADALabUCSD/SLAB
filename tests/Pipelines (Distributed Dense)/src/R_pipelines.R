suppressMessages(library(pbdDMAT))
suppressMessages(library(pbdMPI))
BENCHMARK_PROJECT_ROOT <- Sys.getenv('BENCHMARK_PROJECT_ROOT')
source(paste(BENCHMARK_PROJECT_ROOT, '/lib/R/R_timing_utils.R', sep = ''))
source(paste(BENCHMARK_PROJECT_ROOT, '/lib/R/readDMM.R', sep=''))

init.grid()

nrows <- c(1000,10000,100000,1000000)
colnames <- c('rows', paste('time',1:5,sep=''))
if (comm.rank() == 0) {
    fh <- file('../output/R_pipelines.txt', open='w')
    write(paste(colnames, collapse=','), fh, append=TRUE)
    flush(fh)
} else {
    fh <- NULL
}

for (ix in 1:length(nrows)) {
    r <- nrows[ix]
    t <- ddmatrix("rnorm", nrow=r, ncol=1)
    u <- ddmatrix("rnorm", nrow=1, ncol=r)
    v <- ddmatrix("rnorm", nrow=r, ncol=1)

    times <- double(5)
    for (iter in 1:5) {
        comm.print('SYNCHRONIZING')
        barrier()
        comm.print('START')
        a <- Sys.time()

        res <- t %*% u %*% v
        
        barrier()
        comm.print('COMPLETE')
        b <- Sys.time()
        times[iter] <- as.numeric(b-a, units="secs")
    }
    if (comm.rank() == 0) {
        write(paste(c(r, times), collapse=','), fh, append=TRUE, sep=',')
        flush(fh)
    }
}

finalize()
