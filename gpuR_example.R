library(gpuR)
library(ggplot2)

# Comparacion de tiempos de ejecucion de la operacion multiplicacion de matrices 
# entre la operacion regular (usando el CPU)
# y la operacion implementada por la libreria gpuR (usando la GPU)

#orders <- seq(1000,4000,by=1000)
orders <- seq(1000,2000,by=500)
t1 <- t2 <- t3 <- sizes <- c()

for(order in orders) {
  print(order)
  A = matrix(rnorm(order^2), nrow=order) 
  sizes <- c( sizes, format(object.size(A), units="auto") )

  # GPU version, GPU pointer to CPU memory!! (gpuMatrix is simply a pointer)
  gpuA = gpuMatrix(A, type="float") #point GPU to matrix
  
  # GPU version, in GPU memory!! (vclMatrix formation is a memory transfer)
  vclA = vclMatrix(A, type="float") #push matrix to GPU

  print("cpu...")
  elapsed1 <- system.time({C = A %*% A})[3]  # Operacion regular (ejecutada en la CPU) 
  print("gpu...")
  elapsed2 <- system.time({gpuC = gpuA %*% gpuA})[3] # Operacion acelerada (ejecutada en la GPU)
  print("gpu2...")
  elapsed3 <- system.time({vclC = vclA %*% vclA})[3] # Operacion acelerada (ejecutada en la GPU)
  
  t1 = c(t1, elapsed1)
  t2 = c(t2, elapsed2)
  t3 = c(t3, elapsed3)
  
  # Liberar espacio en memoria:
  rm(A,C,gpuA,gpuC,vclA,vclC); gc()
}

# Armar un dataframe con los resultados del bucle
df = data.frame(order = rep(orders,3), 
                label = rep(paste0(orders," (", sizes, ")"), 3),
                mode = rep(c("cpu","gpu","gpu-2"),each=length(orders)), 
                elapsed = c(t1,t2,t3))

# Guardar el dataframe (ya que el codigo de este chunk para plotearlo a continuación)
saveRDS(df, file = "data/gpuMm.rds")

# Plotear los resultados
ggplot(df, aes(x=label, y=elapsed, fill=mode)) +
  geom_bar(stat="identity", position=position_dodge()) +
  xlab("Orden (Mem)") + 
  ylab("Tiempo de ejecucion (seg)") #+ scale_y_log10() 

# Analisis manual
if(F){
  library(profvis)
  order = 2000
  A = matrix(rnorm(order^2), nrow=order) 
  gpuA = gpuMatrix(A, type="float") 
  
  profvis({C = A %*% A})[3]  # Operacion regular (ejecutada en la CPU) 
  profvis({gpuC = gpuA %*% gpuA})[3] # Operacion acelerada (ejecutada en la GPU)
}