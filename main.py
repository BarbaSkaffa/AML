from chest_xray.features.dataset import XrayDataset

def main():
    manager = XrayManager()
    base_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True)
    ])
    temp_ds = XrayDataset(manager.train_df, transform=base_transform, diseases=manager.diseases)
    temp_loader = DataLoader(temp_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    mean, std = get_normalization_stats(temp_loader)
    print(f"Mean: {mean.item()}, Std: {std.item()}")
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(mean=0.0, sigma=0.02),
        v2.Normalize(mean=[mean.item()], std=[std.item()])
    ])
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[mean.item()], std=[std.item()])
    ])
    folds = manager.get_fold_loaders(k=K_FOLDS, batch_size=BATCH_SIZE, transform=train_transform)
    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        print(f"\n--- Training Fold {fold_idx + 1} ---")
        # In a real training loop, you'd apply val_transform to the val_loader
        # (Small fix: update the val_loader's dataset transform here)
        val_loader.dataset.transform = val_transform


if __name__ == "__main__":
    main()

