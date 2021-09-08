using world size: 8, data-parallel-size: 1, tensor-model-parallel size: 2, pipeline-model-parallel size: 4 
using torch.float16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  bert_binary_head ................................ True
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  checkpoint_activations .......................... True
  checkpoint_num_layers ........................... 1
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_impl ....................................... mmap
  data_parallel_size .............................. 1
  data_path ....................................... ['/workspace/Megatron-LM/my-gpt2_text_document']
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_seq_length .............................. None
  distribute_checkpointed_activations ............. False
  distributed_backend ............................. nccl
  embedding_path .................................. None
  encoder_seq_length .............................. 2048
  eod_mask_loss ................................... False
  eval_interval ................................... 1000
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  ffn_hidden_size ................................. 20480
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  global_batch_size ............................... 1024
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 5120
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_dim ......................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  kv_channels ..................................... 128
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ None
  local_rank ...................................... 0
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 1
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.00015
  lr_decay_iters .................................. 320
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_prob ....................................... 0.15
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 2048
  merge_file ...................................... gpt2-merges.txt
  micro_batch_size ................................ 1
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 40
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_layers ...................................... 40
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  override_lr_scheduler ........................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  pipeline_model_parallel_size .................... 4
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  sample_rate ..................................... 1.0
  save ............................................ None
  save_interval ................................... 10000
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 2048
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 800,100,100
  tensor_model_parallel_size ...................... 2
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  titles_data_path ................................ None
  tokenizer_type .................................. GPT2BPETokenizer
  train_iters ..................................... 1000
  train_samples ................................... None
  use_checkpoint_lr_scheduler ..................... False
  use_contiguous_buffers_in_ddp ................... False
  use_cpu_initialization .......................... None
  use_one_sent_docs ............................... False
  virtual_pipeline_model_parallel_size ............ None
  vocab_extra_ids ................................. 0
  vocab_file ...................................... gpt2-vocab.json
  weight_decay .................................... 0.01
  world_size ...................................... 8
-------------------- end of arguments ---------------------
setting number of micro-batches to constant 1024
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 175 dummy tokens (new size: 50432)
> initializing torch distributed ...
> initializing tensor model parallel with size 2
> initializing pipeline model parallel with size 4
> setting random seeds to 1234 ...
> initializing model parallel cuda seeds on global rank 0, model parallel rank 0, and data parallel rank 0 with model parallel seed: 3952 and data parallel seed: 1234
> compiling dataset index builder ...
make: Entering directory '/workspace/Megatron-LM/megatron/data'
make: Nothing to be done for 'default'.
make: Leaving directory '/workspace/Megatron-LM/megatron/data'
>>> done with dataset index builder. Compilation time: 0.064 seconds
> compiling and loading fused kernels ...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_upper_triang_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_upper_triang_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module scaled_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /workspace/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module fused_mix_prec_layer_norm_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module fused_mix_prec_layer_norm_cuda...
>>> done with compiling and loading fused kernels. Compilation time: 1.150 seconds
time to initialize megatron (seconds): 41.282
[after megatron is initialized] datetime: 2021-09-08 03:19:05 
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (0, 1): 1573350400
 > number of parameters on (tensor, pipeline) model parallel rank (1, 2): 1573350400
 > number of parameters on (tensor, pipeline) model parallel rank (1, 1): 1573350400
 > number of parameters on (tensor, pipeline) model parallel rank (0, 2): 1573350400
 > number of parameters on (tensor, pipeline) model parallel rank (0, 3): 1702466560
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 1712942080
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 1712942080
 > number of parameters on (tensor, pipeline) model parallel rank (1, 3): 1702466560
> learning rate decay style: cosine
[after model, optimizer, and learning rate scheduler are built] datetime: 2021-09-08 03:19:06 
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      1024000
    validation: 20480
    test:       10240
> building train, validation, and test datasets for GPT ...
 > building dataset index ...
    reading sizes...
    reading pointers...
    reading document index...
    creating numpy buffer of mmap...
    creating memory view of numpy buffer...
 > finished creating indexed dataset in 0.000868 seconds
    number of documents: 398
 > dataset split:
    train:
     document indices in [0, 318) total of 318 documents
    validation:
     document indices in [318, 358) total of 40 documents
    test:
     document indices in [358, 398) total of 40 documents
 > loading doc-idx mapping from /workspace/Megatron-LM/my-gpt2_text_document_train_indexmap_1024000ns_2048sl_1234s_doc_idx.npy
 > loading sample-idx mapping from /workspace/Megatron-LM/my-gpt2_text_document_train_indexmap_1024000ns_2048sl_1234s_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/Megatron-LM/my-gpt2_text_document_train_indexmap_1024000ns_2048sl_1234s_shuffle_idx.npy
    loaded indexed file in 0.001 seconds
    total number of samples: 1024001
    total number of epochs: 1097986
 > loading doc-idx mapping from /workspace/Megatron-LM/my-gpt2_text_document_valid_indexmap_20480ns_2048sl_1234s_doc_idx.npy
 > loading sample-idx mapping from /workspace/Megatron-LM/my-gpt2_text_document_valid_indexmap_20480ns_2048sl_1234s_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/Megatron-LM/my-gpt2_text_document_valid_indexmap_20480ns_2048sl_1234s_shuffle_idx.npy
    loaded indexed file in 0.001 seconds
    total number of samples: 20481
    total number of epochs: 174763
 > loading doc-idx mapping from /workspace/Megatron-LM/my-gpt2_text_document_test_indexmap_10240ns_2048sl_1234s_doc_idx.npy
 > loading sample-idx mapping from /workspace/Megatron-LM/my-gpt2_text_document_test_indexmap_10240ns_2048sl_1234s_sample_idx.npy
 > loading shuffle-idx mapping from /workspace/Megatron-LM/my-gpt2_text_document_test_indexmap_10240ns_2048sl_1234s_shuffle_idx.npy
    loaded indexed file in 0.001 seconds
    total number of samples: 10241
    total number of epochs: 87382
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2021-09-08 03:19:09 
done with setup ...
time (ms) | model-and-optimizer-setup: 971.92 | train/valid/test-data-iterators-setup: 3361.10
training ...
[before the start of training step] datetime: 2021-09-08 03:19:09 
 iteration        1/    1000 | consumed samples:         1024 | elapsed time per iteration (ms): 201212.4 | learning rate: 0.000E+00 | global batch size:  1024 | loss scale: 4294967296.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
time (ms) | forward-compute: 51842.73 | forward-recv: 5954.23 | backward-compute: 136988.70 | backward-send: 0.25 | backward-send-forward-recv: 1926.01 | backward-params-all-reduce: 29.54 | backward-embedding-all-reduce: 446.99 | optimizer-copy-to-main-grad: 18.71 | optimizer-unscale-and-check-inf: 3877.65 | optimizer: 3896.49 | batch-generator: 535.53
 iteration        2/    1000 | consumed samples:         2048 | elapsed time per iteration (ms): 188590.1 | learning rate: 0.000E+00 | global batch size:  1024 | loss scale: 2147483648.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
time (ms) | forward-compute: 50539.84 | forward-recv: 129.21 | backward-compute: 136692.63 | backward-send: 0.25 | backward-send-forward-recv: 668.63 | backward-params-all-reduce: 29.92 | backward-embedding-all-reduce: 381.36 | optimizer-copy-to-main-grad: 9.63 | optimizer-unscale-and-check-inf: 13.34 | optimizer: 24.18 | batch-generator: 523.89
 iteration        3/    1000 | consumed samples:         3072 | elapsed time per iteration (ms): 188879.7 | learning rate: 0.000E+00 | global batch size:  1024 | loss scale: 1073741824.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
time (ms) | forward-compute: 50600.13 | forward-recv: 126.48 | backward-compute: 136934.03 | backward-send: 0.24 | backward-send-forward-recv: 668.14 | backward-params-all-reduce: 20.22 | backward-embedding-all-reduce: 381.49 | optimizer-copy-to-main-grad: 9.61 | optimizer-unscale-and-check-inf: 13.28 | optimizer: 23.02 | batch-generator: 520.28
 iteration        4/    1000 | consumed samples:         4096 | elapsed time per iteration (ms): 190863.1 | learning rate: 0.000E+00 | global batch size:  1024 | loss scale: 536870912.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
time (ms) | forward-compute: 51055.60 | forward-recv: 126.72 | backward-compute: 138457.12 | backward-send: 0.26 | backward-send-forward-recv: 674.70 | backward-params-all-reduce: 20.27 | backward-embedding-all-reduce: 381.15 | optimizer-copy-to-main-grad: 9.62 | optimizer-unscale-and-check-inf: 13.18 | optimizer: 22.92 | batch-generator: 521.97
 iteration        5/    1000 | consumed samples:         5120 | elapsed time per iteration (ms): 192769.9 | learning rate: 0.000E+00 | global batch size:  1024 | loss scale: 268435456.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
time (ms) | forward-compute: 51736.40 | forward-recv: 126.78 | backward-compute: 139663.52 | backward-send: 0.26 | backward-send-forward-recv: 680.60 | backward-params-all-reduce: 20.30 | backward-embedding-all-reduce: 392.02 | optimizer-copy-to-main-grad: 9.62 | optimizer-unscale-and-check-inf: 13.36 | optimizer: 23.10 | batch-generator: 525.28
 iteration        6/    1000 | consumed samples:         6144 | elapsed time per iteration (ms): 192465.4 | learning rate: 0.000E+00 | global batch size:  1024 | loss scale: 134217728.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
time (ms) | forward-compute: 51707.88 | forward-recv: 126.24 | backward-compute: 139384.79 | backward-send: 0.25 | backward-send-forward-recv: 681.88 | backward-params-all-reduce: 20.57 | backward-embedding-all-reduce: 394.34 | optimizer-copy-to-main-grad: 9.62 | optimizer-unscale-and-check-inf: 13.52 | optimizer: 23.27 | batch-generator: 526.94
 iteration        7/    1000 | consumed samples:         7168 | elapsed time per iteration (ms): 192667.9 | learning rate: 0.000E+00 | global batch size:  1024 | loss scale: 67108864.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
time (ms) | forward-compute: 51758.33 | forward-recv: 127.39 | backward-compute: 139537.65 | backward-send: 0.26 | backward-send-forward-recv: 680.56 | backward-params-all-reduce: 20.51 | backward-embedding-all-reduce: 393.13 | optimizer-copy-to-main-grad: 9.61 | optimizer-unscale-and-check-inf: 13.37 | optimizer: 23.10 | batch-generator: 525.63
 iteration        8/    1000 | consumed samples:         8192 | elapsed time per iteration (ms): 192854.6 | learning rate: 0.000E+00 | global batch size:  1024 | loss scale: 33554432.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
time (ms) | forward-compute: 51817.40 | forward-recv: 127.06 | backward-compute: 139665.98 | backward-send: 0.26 | backward-send-forward-recv: 681.25 | backward-params-all-reduce: 20.43 | backward-embedding-all-reduce: 393.54 | optimizer-copy-to-main-grad: 9.61 | optimizer-unscale-and-check-inf: 13.29 | optimizer: 23.02 | batch-generator: 526.93
 iteration        9/    1000 | consumed samples:         9216 | elapsed time per iteration (ms): 192785.9 | learning rate: 0.000E+00 | global batch size:  1024 | loss scale: 16777216.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
time (ms) | forward-compute: 51772.46 | forward-recv: 126.71 | backward-compute: 139639.28 | backward-send: 0.26 | backward-send-forward-recv: 683.42 | backward-params-all-reduce: 20.50 | backward-embedding-all-reduce: 393.65 | optimizer-copy-to-main-grad: 9.62 | optimizer-unscale-and-check-inf: 12.81 | optimizer: 22.56 | batch-generator: 527.61
 iteration       10/    1000 | consumed samples:        10240 | elapsed time per iteration (ms): 192930.3 | learning rate: 0.000E+00 | global batch size:  1024 | loss scale: 8388608.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
time (ms) | forward-compute: 51866.22 | forward-recv: 126.86 | backward-compute: 139692.51 | backward-send: 0.25 | backward-send-forward-recv: 680.99 | backward-params-all-reduce: 20.41 | backward-embedding-all-reduce: 394.36 | optimizer-copy-to-main-grad: 9.62 | optimizer-unscale-and-check-inf: 13.01 | optimizer: 22.77 | batch-generator: 526.59
