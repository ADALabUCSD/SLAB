makePlot <- function(data, xVarName, 
                     title='', xlab='', 
                     ylab='', saving=NULL,
                     epsScale = .01, errorBars=TRUE,
                     legendPos = 'topleft',
                     legend=FALSE, excludeSystems=c()) {
    
    yAxisLimits <- getYAxisLimits(data, errorBars)
    xAxisLimits <- c(min(data[,xVarName]), max(data[,xVarName]))
    
    if (!is.null(saving)) {
        png(saving)
    }
    plot(x=1, y=1, type='n', xlim = xAxisLimits, 
         ylim = yAxisLimits, main = title,
         xlab = xlab, ylab = ylab)
    
    xData <- data[,xVarName]
    eps <- epsScale*(abs(max(xData)-min(xData)))
    
    params <- findPresentSystems(names(data))
    for (p in params) {
        if (p$name %in% excludeSystems) next
        timeName <- paste('time_', p$name, sep='')
        varName  <- paste('variance_', p$name, sep='')
        
        time <- data[,timeName]
        timeUpper <- time + 2*sqrt(data[,varName])
        timeLower <- time - 2*sqrt(data[,varName])
        
        lines(x=xData, y=time, col=p$plottingColor, lty=p$lineType)
        
        if (errorBars) {
            segments(x0=xData-eps, y0=timeUpper, x1=xData+eps, y1=timeUpper)
            segments(x0=xData-eps, y0=timeLower, x1=xData+eps, y1=timeLower)
            segments(x0=xData, y0=timeLower, x1=xData, y1=timeUpper)
        }
    }
    
    if (legend) {
        prettyNames <- unlist(lapply(params, FUN='[[', 'prettyName'))
        lineTypes <- unlist(lapply(params, FUN='[[', 'lineType'))
        colors <- unlist(lapply(params, FUN='[[', 'plottingColor'))
        
        legend(x = legendPos, legend = prettyNames,
               lty = lineTypes, col = colors, bty = 'n')
    }
    if (!is.null(saving)) {
        dev.off()
    }
}

getYAxisLimits <- function(data, errorBars) {
    globalMax <- 0
    globalMin <- 0
    timeCols <- grep('time', names(data), value=TRUE)
    for (timeName in timeCols) {
        varName <- sub('time','variance', timeName)
        thisMax <- max(data[,timeName], na.rm = TRUE)
        thisMin <- min(data[,timeName], na.rm = TRUE)
        if (errorBars) { 
            thisMax <- max(data[,timeName] + 2*sqrt(data[,varName]), na.rm=TRUE)
        }
        if (errorBars) {
            thisMin <- min(data[,timeName] - 2*sqrt(data[,varName]), na.rm=TRUE)
        }
        if (thisMax > globalMax) globalMax <- thisMax
        if (thisMin < globalMin) globalMin <- thisMin
    }
    upper <- globalMax + .05*globalMax
    lower <- globalMin - .05*globalMin
    limits <- c(lower, upper)
    return(limits)
}

getData <- function(dirName, filterText=NULL, 
                    excludeSystems=c(),
                    excludeText=c(),
                    excludeCols=c()) {
    files <- grep('.txt', dir(dirName), value=TRUE)
    if (!is.null(filterText)) {
        for (cond in filterText) {
            pattern <- paste('_', cond, sep='')
            files <- grep(pattern, files, value=TRUE)
        }
    }
    
    for (filter in excludeText) {
        files <- files[!grepl(filter, files)]    
    }
    
    files <- paste(dirName, '/', files, sep='')
    params <- findPresentSystems(files)
    
    baseName <- params[[1]]$name
    baseFile <- grep(paste(baseName, '_', sep=''), files, value=TRUE)
    if (length(baseFile) == 0) {
        baseName <- params[[1]]$alias
        baseFile <- grep(baseName, files, value=TRUE)
    }
    
    data <- read.csv(baseFile)
    excludeColIx <- which(names(data) %in% excludeCols)
    if (length(excludeColIx) != 0) {
        data <- data[,-excludeColIx]
    }
    dataVars <- which(names(data) == 'time' | names(data) == 'variance')
    joinVars <- names(data)[-dataVars]
    data <- data[,c(joinVars, names(data)[dataVars])]
    dataVars <- which(names(data) == 'time' | names(data) == 'variance')
    names(data) <- c(joinVars, paste(names(data)[dataVars], '_', baseName, sep=''))
    for (p in params) {
        if ((p$name == baseName) || (p$alias == baseName)) next
        if (p$name %in% excludeSystems) next
        pattern <- paste(p$name, '_', sep='')
        filename <- grep(pattern, files, value=TRUE)
        if (length(filename) == 0) {
            pattern <- paste(p$alias, '_', sep='')
            filename <- grep(pattern, files, value=TRUE)
        }
        if (length(filename) == 0) {
            warning(paste('No file found for system', p$name))
            next
        }
        
        slice <- read.csv(filename)
        excludeColIx <- which(names(slice) %in% excludeCols)
        if (length(excludeColIx) != 0) {
            slice <- slice[,-excludeColIx]
        }
        data <- merge(data, slice, joinVars, all = TRUE)
        names(data)[names(data) == 'time'] = paste('time_', p$name, sep='')
        names(data)[names(data) == 'variance'] = paste('variance_', 
                                                       p$name, sep='')
    }
    return(data)
}

findPresentSystems <- function(searchString) {
    params <- getSystemParams()
    systems <- list()
    ix <- 1
    for (p in params) {
        pattern <- paste(p$name, '_', sep='')
        is_present <- any(grepl(pattern, searchString))
        if (!is_present) {
            pattern <- paste('_', p$name, sep='')
            is_present <- any(grepl(pattern, searchString))
        }
        if ((!is_present) && (p$alias != '')) {
            pattern <- paste(p$alias, '_', sep='')
            is_present <- any(grepl(pattern, searchString))
        }
        if ((!is_present) && (p$alias != '')) {
            pattern <- paste('_', p$alias, sep='')
            is_present <- any(grepl(pattern, searchString))
        }
        if (is_present) {
            systems[[ix]] <- p
            ix <- ix+1
        }
    }
    
    return(systems)
}

getSystemParams <- function() {
    systems <- c('R','np','tf',
                 'sysml','madlib',
                 'samsara','blas_double','blas_float', 'eigen_double',
                 'eigen_float', 'apache_array', 'apache_block', 'mllib')
    prettyNames <- c('R', 'Numpy', 'TensorFlow', 'SystemML', 
                     'MADLib', 'Samsara', 'BLAS (Doubles)', 'BLAS (Floats)',
                     'Eigen3 (Doubles)', 'Eigen3 (Floats)',
                     'Commons Math (Arrays)', 'Commons Math (Blocks)',
                     'Spark MLLib')
    aliases <- c('','','','systemml','','','','','','','','','') 
    plottingColors <- c('red','lawngreen','turquoise','purple',
                        'indianred','midnightblue','orange','orange4',
                        'orchid','springgreen4','steelblue4','slategray',
                        'mediumblue')
    
    params <- vector('list', length(systems))
    for (ix in 1:length(systems)) {
        params[[ix]] <- list(name = systems[ix],
                             prettyName = prettyNames[ix],
                             alias = aliases[ix],
                             plottingColor = plottingColors[ix],
                             lineType = ix)
    }
    return(params)
}
