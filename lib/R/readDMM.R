capture.output(library(pbdMPI, quietly=TRUE))
capture.output(library(pbdDMAT, quietly=TRUE))

# Thanks to the PBDR Package (see citation in paper)
# for the skeleton of this code

readDMM <- function(path, bldim=NULL, num_readers=10, ICTXT=0, sep=',') {
    dims <- capture.output(parseMetadata(path))
    rows <- dims[['rows']]
    cols <- dims[['cols']]

    if (is.null(bldim)) {
        bldim <- 32
    }
    comm.print(paste('Blocking dimension: ', bldim, sep=''))
    CTXT <- base.minctxt()
    blacs_gridinit(ICTXT=CTXT, NPROW=num_readers, NPCOL=1L)

    blacs_ <- base.blacs(CTXT)

    rows_to_read <- ceiling(rows / num_readers)
    comm.print('READING...')
    if (blacs_$MYROW != -1) {
        lines_to_skip <- comm.rank() * rows_to_read
        chunk <- scan(file=path, skip=lines_to_skip,
                      nlines=rows_to_read, sep=sep, quiet=TRUE)
    } else {
        chunk <- NULL
    }
    if (is.null(chunk) || length(chunk) == 0L) {
        submat <- matrix(0)
    } else {
        submat <- matrix(chunk, ncol=cols, byrow=TRUE)
    }

    local_dim <- dim(submat)
    blocking_dim <- c(rows_to_read, cols)
    M <- new("ddmatrix", Data=submat, dim=c(rows,cols),
             ldim=local_dim, bldim=blocking_dim, ICTXT=CTXT)
    if (length(bldim) == 1) {
        bldim <- rep(bldim,2)
    }
    comm.print('REDISTRIBUTING...')
    M <- pbdDMAT::redistribute(dx=M, bldim=bldim, ICTXT=ICTXT)
    gridexit(CTXT)
    return(M)
}

parseMetadata <- function(path) {
    data <- scan(paste(path, '.mtd', sep=''), what=character(), quiet=TRUE)
    col_ix <- which(data == 'cols')
    row_ix <- which(data == 'rows')

    dims <- list()
    dims[['rows']] = as.integer(sub(',', '', data[row_ix+2]))
    dims[['cols']] = as.integer(sub(',', '', data[col_ix+2]))


    return(dims)
}
