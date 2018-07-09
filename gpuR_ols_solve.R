# OLS solve
# Ordinary Least Squares

set.seed(123456)
np <- 300  #number of predictors (Antes: 30)
nr <- 1e+05  #number of observations
X <- cbind(5, 1:nr, matrix(rnorm((np - 1) * nr, 0, 0.01), nrow = nr, ncol = (np - 1)))
beta <- matrix(c(1, 3, runif(np - 1, 0, 0.2)), ncol = 1)
y <- X %*% beta + matrix(rnorm(nr, 0, 1), nrow = nr, ncol = 1)

# CPU bound version, slight optimize via crossprod but otherwise vanilla
timeCpu <- system.time({
  ms2 <- solve(crossprod(X), crossprod(X, y))
})

# GPU version, GPU pointer to CPU memory!! (gpuMatrix is simply a pointer)
library(gpuR)
gpuX = gpuMatrix(X, type = "float")  #point GPU to matrix
gpuy = gpuMatrix(y, type = "float")
timeGpu <- system.time({
  ms4 <- gpuR::solve(gpuR::crossprod(gpuX), gpuR::crossprod(gpuX, gpuy))
})

# GPU version, in GPU memory!! (vclMatrix formation is a memory transfer)
vclX = vclMatrix(X, type = "float")  #push matrix to GPU
vcly = vclMatrix(y, type = "float")
timeGpu2 <- system.time({
  ms5 <- gpuR::solve(gpuR::crossprod(vclX), gpuR::crossprod(vclX, vcly))
})

detach("package:gpuR", unload = TRUE)

# Armar un dataframe con los resultados del bucle
df = data.frame(calculo = c("CPU", "GPU (gpu*)", "GPU (vcl*)"), 
                tiempo = c(timeCpu[3], timeGpu[3], timeGpu2[3]) )
saveRDS(df, file = "data/gpuOLS.rds")

# Plotear los resultados
ggplot(df, aes(x=calculo, y=tiempo, fill=calculo)) +
  geom_bar(stat="identity", position=position_dodge()) +
  xlab("Tipo de calculo") + 
  ylab("Tiempo de ejecucion (seg)") #+ scale_y_log10() 
