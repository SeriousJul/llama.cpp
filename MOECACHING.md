# moe caching

```bash
set -e

REPO="$HOME/.cache/huggingface/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF"
BACKUP="${REPO}.pre-ext4-backup"
IMAGE="$HOME/.cache/huggingface/hub/DeepSeek-V4-Flash-0731-GGUF.ext4.img"

# Check the current repository size first.
du -sh "$REPO"

# Keep the original repository as a rollback copy.
mv "$REPO" "$BACKUP"

# Create a sparse 140 GiB virtual disk.
truncate -s 140G "$IMAGE"

# Format it as ext4.
mkfs.ext4 -F -m 0 "$IMAGE"

# Recreate the original mount point.
mkdir -p "$REPO"

# Mount the virtual ext4 filesystem.
sudo mount -o loop,noatime "$IMAGE" "$REPO"

# Copy the Hugging Face repository into the ext4 filesystem.
cp -a "$BACKUP"/. "$REPO"/

# Make the copied files usable by your user.
sudo chown -R "$USER:$(id -gn)" "$REPO"

# Confirm the mount and filesystem.
findmnt -T "$REPO"
stat -f -c 'filesystem=%T' "$REPO"
du -sh "$REPO"
```

Then retry:

```bash
llama serve \
 -hf unsloth/DeepSeek-V4-Flash-0731-GGUF:IQ3_XXS \
 --load-mode mmap \
 --moe-cache generation \
 --moe-cache-vram 8192 \
 --moe-cache-ram 49152 \
 --moe-cache-host-reserve 12288 \
 --fit on
```

The image file must be stored on a disk with at least enough free space for the copied model. truncate creates a sparse file, but copying the model will allocate real storage.

Verify direct I/O against one shard:

```bash
SHARD=$(find "$REPO/snapshots" -type f -name '*.gguf' | sort | head -n 1)
dd if="$SHARD" of=/dev/null bs=4096 count=1 iflag=direct status=none
echo "direct I/O succeeded"
```

After confirming the model loads, remove the old copy only if you no longer need rollback:

```bash
rm -rf "$BACKUP"
```

To unmount later:

```bash
sudo umount "$REPO"
```

To restore the original repository:

```bash
sudo rm -rf "$REPO"
mv "$BACKUP" "$REPO"
```
