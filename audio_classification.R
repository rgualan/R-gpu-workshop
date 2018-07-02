# Modeling
library(keras)
library(tfruns)


# Downloading
dir.create("data")

download.file(
  url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz", 
  destfile = "data/speech_commands_v0.01.tar.gz"
)

untar("data/speech_commands_v0.01.tar.gz", exdir = "data/speech_commands_v0.01")



# Importing
library(stringr)
library(dplyr)

files <- fs::dir_ls(
  path = "data/speech_commands_v0.01/", 
  recursive = TRUE, 
  glob = "*.wav"
)

files <- files[!str_detect(files, "background_noise")]

df <- data_frame(
  fname = files, 
  class = fname %>% str_extract("1/.*/") %>% 
    str_replace_all("1/", "") %>%
    str_replace_all("/", ""),
  class_id = class %>% as.factor() %>% as.integer() - 1L
)


# Generator
library(tfdatasets)
ds <- tensor_slices_dataset(df) 

window_size_ms <- 30
window_stride_ms <- 10

window_size <- as.integer(16000*window_size_ms/1000)
stride <- as.integer(16000*window_stride_ms/1000)

fft_size <- as.integer(2^trunc(log(window_size, 2)) + 1)
n_chunks <- length(seq(window_size/2, 16000 - window_size/2, stride))

# shortcuts to used TensorFlow modules.
audio_ops <- tf$contrib$framework$python$ops$audio_ops

ds <- ds %>%
  dataset_map(function(obs) {
    
    # a good way to debug when building tfdatsets pipelines is to use a print
    # statement like this:
    # print(str(obs))
    
    # decoding wav files
    audio_binary <- tf$read_file(tf$reshape(obs$fname, shape = list()))
    wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1)
    
    # create the spectrogram
    spectrogram <- audio_ops$audio_spectrogram(
      wav$audio, 
      window_size = window_size, 
      stride = stride,
      magnitude_squared = TRUE
    )
    
    # normalization
    spectrogram <- tf$log(tf$abs(spectrogram) + 0.01)
    
    # moving channels to last dim
    spectrogram <- tf$transpose(spectrogram, perm = c(1L, 2L, 0L))
    
    # transform the class_id into a one-hot encoded vector
    response <- tf$one_hot(obs$class_id, 30L)
    
    list(spectrogram, response)
  }) 


ds <- ds %>% 
  dataset_shuffle(buffer_size = 100) %>%
  dataset_repeat() %>%
  dataset_padded_batch(
    batch_size = 32, 
    padded_shapes = list(
      shape(n_chunks, fft_size, NULL), 
      shape(NULL)
    )
  )


data_generator <- function(df, batch_size, shuffle = TRUE, 
                           window_size_ms = 30, window_stride_ms = 10) {
  
  window_size <- as.integer(16000*window_size_ms/1000)
  stride <- as.integer(16000*window_stride_ms/1000)
  fft_size <- as.integer(2^trunc(log(window_size, 2)) + 1)
  n_chunks <- length(seq(window_size/2, 16000 - window_size/2, stride))
  
  ds <- tensor_slices_dataset(df)
  
  if (shuffle) 
    ds <- ds %>% dataset_shuffle(buffer_size = 100)  
  
  ds <- ds %>%
    dataset_map(function(obs) {
      
      # decoding wav files
      audio_binary <- tf$read_file(tf$reshape(obs$fname, shape = list()))
      wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1)
      
      # create the spectrogram
      spectrogram <- audio_ops$audio_spectrogram(
        wav$audio, 
        window_size = window_size, 
        stride = stride,
        magnitude_squared = TRUE
      )
      
      spectrogram <- tf$log(tf$abs(spectrogram) + 0.01)
      spectrogram <- tf$transpose(spectrogram, perm = c(1L, 2L, 0L))
      
      # transform the class_id into a one-hot encoded vector
      response <- tf$one_hot(obs$class_id, 29L)
      
      list(spectrogram, response)
    }) %>%
    dataset_repeat()
  
  ds <- ds %>% 
    dataset_padded_batch(batch_size, list(shape(n_chunks, fft_size, NULL), shape(NULL)))
  
  ds
}


set.seed(6)
id_train <- sample(nrow(df), size = 0.7*nrow(df))

ds_train <- data_generator(
  df[id_train,], 
  batch_size = 32, 
  window_size_ms = 30, 
  window_stride_ms = 10
)

ds_validation <- data_generator(
  df[-id_train,], 
  batch_size = 32, 
  shuffle = FALSE, 
  window_size_ms = 30, 
  window_stride_ms = 10
)

sess <- tf$Session()
batch <- next_batch(ds_train)
str(sess$run(batch))


# Model definition

window_size <- as.integer(16000*window_size_ms/1000)
stride <- as.integer(16000*window_stride_ms/1000)
fft_size <- as.integer(2^trunc(log(window_size, 2)) + 1)
n_chunks <- length(seq(window_size/2, 16000 - window_size/2, stride))


model <- keras_model_sequential()
model %>%  
  layer_conv_2d(input_shape = c(n_chunks, fft_size, 1), 
                filters = 32, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 256, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 29, activation = 'softmax')

model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Model fitting
model %>% fit_generator(
  generator = ds_train,
  steps_per_epoch = 0.7*nrow(df)/32,
  epochs = 10, 
  validation_data = ds_validation, 
  validation_steps = 0.3*nrow(df)/32
)
# Nota:
# El entrenamiento es muy lento usando el CPU, aproximadamente 1 segundo por step? x/1415
# Espero que esto sea mucho más rápido al usar el GPU




