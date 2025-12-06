"""
Improved Early Stopping Configuration for Training Loop

This script provides enhanced early stopping logic with additional overfitting monitoring.
Copy this code into your notebook's Training Loop cell to replace lines 295 and 358-369.

Key improvements:
- Reduced patience from 7 to 5 (stop earlier when overfitting detected)
- Added train/val loss gap monitoring
- Added warning when gap exceeds threshold
- More detailed logging of training progress
"""

# ======================
# TRAINING CONFIGURATION (Replace line 295)
# ======================

epochs = 30
patience = 5  # Reduced from 7 - stop earlier when overfitting detected
min_delta = 0.01  # Minimum improvement to count as progress
best_val = float("inf")
bad = 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

output_dir = "../runs"
model_dir = "../models"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

metrics_file = os.path.join(output_dir, "metrics_v2.json")
model_save_path = os.path.join(model_dir, "brain_tumor_resnet18_v2_trained.pt")

print("Starting training...")

# ======================
# IMPROVED EARLY STOPPING LOGIC (Replace lines 358-369)
# ======================

# Note: Place this code inside your training loop, after calculating val_loss and val_acc
# It should replace the existing early stopping logic

    # Enhanced Early Stopping with overfitting monitoring
    loss_gap = train_loss - val_loss

    # Check for improvement (with minimum delta threshold)
    if val_loss < (best_val - min_delta):
        best_val = val_loss
        best_state = deepcopy(model.state_dict())
        torch.save(best_state, model_save_path)
        print(f"  → Model saved to {model_save_path}")
        bad = 0
    else:
        bad += 1

    # Warning if overfitting detected
    if loss_gap > 0.3:
        print(f"  ⚠️  Overfitting detected: train/val loss gap = {loss_gap:.4f}")

    # Early stopping
    if bad >= patience:
        print(f"Early stopping after {epoch + 1} epochs (no improvement for {patience} epochs)")
        break

print("\nTraining complete!")

# ======================
# COMPLETE TRAINING LOOP EXAMPLE
# ======================

"""
Here's the complete training loop with all improvements integrated:

for epoch in range(epochs):
    # --- Training ---
    model.train()
    tl, tc, tt = 0.0, 0, 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        tl += loss.item() * x.size(0)
        tc += (out.argmax(1) == y).sum().item()
        tt += y.size(0)

    train_loss = tl / tt
    train_acc = 100 * tc / tt

    # --- Validation ---
    model.eval()
    vl, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            vl += loss.item() * x.size(0)
            vc += (out.argmax(1) == y).sum().item()
            vt += y.size(0)

    val_loss = vl / vt
    val_acc = 100 * vc / vt

    # Scheduler Step
    scheduler.step(val_loss)

    # Save history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch + 1:02d} | Train {train_loss:.4f}, Acc {train_acc:.2f}% | Val {val_loss:.4f}, Acc {val_acc:.2f}%"
    )

    # Enhanced Early Stopping with overfitting monitoring
    loss_gap = train_loss - val_loss

    if val_loss < (best_val - min_delta):
        best_val = val_loss
        best_state = deepcopy(model.state_dict())
        torch.save(best_state, model_save_path)
        print(f"  → Model saved to {model_save_path}")
        bad = 0
    else:
        bad += 1

    if loss_gap > 0.3:
        print(f"  ⚠️  Overfitting detected: train/val loss gap = {loss_gap:.4f}")

    if bad >= patience:
        print(f"Early stopping after {epoch + 1} epochs (no improvement for {patience} epochs)")
        break

print("\nTraining complete!")
"""
