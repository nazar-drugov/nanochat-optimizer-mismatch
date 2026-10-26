# Next changes

- Add distributed support for AdamW before running multi-GPU AdamW experiments. Plain `torch.optim.AdamW` works for one GPU, but nanoChat does not use standard PyTorch DDP to synchronize gradients. Add gradient synchronization and a multi-GPU test.
