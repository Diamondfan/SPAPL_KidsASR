trainer.py

line 836:
if not isinstance(train_dataset, torch.utils.data.IterableDataset):
    dataloader_params["sampler"] = self._get_train_sampler()
    dataloader_params["drop_last"] = self.args.dataloader_drop_last
    dataloader_params["worker_init_fn"] = seed_worker
    return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

if isinstance(train_dataset, torch.utils.data.IterableDataset):
    return DataLoader(train_dataset, **dataloader_params)



line 1792

if hasattr(epoch_iterator.dataset, "set_epoch"):
    epoch_iterator.dataset.set_epoch(epoch)

