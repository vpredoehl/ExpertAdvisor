#!/bin/bash

#
# Full local -> network repository backup.
#
# Source:
#   /Volumes/Developer/ExpertAdvisor
#     Full Git history and full Git LFS object store.
#
# Target:
#   /Volumes/TC3T HD/ExpertAdvisor.git
#     Bare mirror containing full Git history and full Git LFS object store.
#
# This script does NOT rewrite Git history and does NOT filter database
# backups.  The network repository is intended to preserve the complete
# local repository, including all historical LFS database objects.
#

set -u

SOURCE_REPO="/Volumes/Developer/ExpertAdvisor"
TARGET_REPO="/Volumes/TC3T HD/ExpertAdvisor.git"
KEEP_BRANCH="refs/heads/lstm-feature-development"

TARGET_PARENT="$(dirname "$TARGET_REPO")"
TARGET_NAME="$(basename "$TARGET_REPO")"

NEW_TARGET="$TARGET_PARENT/.${TARGET_NAME}.new.$$"
OLD_TARGET="$TARGET_PARENT/.${TARGET_NAME}.old.$$"

fail()
{
    echo
    echo "ERROR: $*" >&2
    exit 1
}

cleanup()
{
    if [ -d "$NEW_TARGET" ]; then
        rm -rf "$NEW_TARGET"
    fi
}

trap cleanup EXIT

echo "=== FULL NETWORK REPOSITORY BACKUP ==="
echo
echo "Source: $SOURCE_REPO"
echo "Target: $TARGET_REPO"
echo

echo "=== PRE-CHECKS ==="

command -v git >/dev/null 2>&1 ||
    fail "git is not available"

command -v rsync >/dev/null 2>&1 ||
    fail "rsync is not available"

git lfs version >/dev/null 2>&1 ||
    fail "Git LFS is not available"

[ -d "$SOURCE_REPO/.git" ] ||
    fail "Source repository does not exist: $SOURCE_REPO"

[ -d "$TARGET_PARENT" ] ||
    fail "Network volume is not mounted: $TARGET_PARENT"

git -C "$SOURCE_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
    fail "Source is not a valid Git worktree"

git -C "$SOURCE_REPO" show-ref --verify --quiet "$KEEP_BRANCH" ||
    fail "Source does not contain $KEEP_BRANCH"

SOURCE_HEAD="$(
    git -C "$SOURCE_REPO" rev-parse "$KEEP_BRANCH"
)" || fail "Could not determine source branch tip"

SOURCE_SYMBOLIC_HEAD="$(
    git -C "$SOURCE_REPO" symbolic-ref HEAD 2>/dev/null
)" || fail "Could not determine source HEAD"

echo "Source HEAD: $SOURCE_SYMBOLIC_HEAD"
echo "Source tip:  $SOURCE_HEAD"

echo
echo "=== VERIFY SOURCE GIT ==="

git -C "$SOURCE_REPO" fsck --full ||
    fail "Source Git repository failed fsck"

echo
echo "=== VERIFY SOURCE LFS ==="

git -C "$SOURCE_REPO" lfs fsck ||
    fail "Source Git LFS store failed fsck"

SOURCE_LFS_DIR="$SOURCE_REPO/.git/lfs/objects"

[ -d "$SOURCE_LFS_DIR" ] ||
    fail "Source Git LFS object directory does not exist"

SOURCE_LFS_COUNT="$(
    find "$SOURCE_LFS_DIR" -type f | wc -l | tr -d ' '
)"

SOURCE_LFS_BYTES="$(
    find "$SOURCE_LFS_DIR" -type f -exec stat -f '%z' {} \; |
        awk '{sum += $1} END {printf "%.0f\n", sum}'
)"

echo "Source LFS objects: $SOURCE_LFS_COUNT"
echo "Source LFS bytes:   $SOURCE_LFS_BYTES"

echo
echo "=== BUILD FRESH BARE MIRROR ==="

rm -rf "$NEW_TARGET"

GIT_LFS_SKIP_SMUDGE=1 \
git clone --mirror \
    "$SOURCE_REPO" \
    "$NEW_TARGET" ||
    fail "Could not create temporary network mirror"

echo
echo "=== COPY COMPLETE LFS STORE ==="

mkdir -p "$NEW_TARGET/lfs/objects" ||
    fail "Could not create target LFS object directory"

rsync -a \
    "$SOURCE_LFS_DIR/" \
    "$NEW_TARGET/lfs/objects/" ||
    fail "Could not copy Git LFS object store"

TARGET_LFS_COUNT="$(
    find "$NEW_TARGET/lfs/objects" -type f | wc -l | tr -d ' '
)"

TARGET_LFS_BYTES="$(
    find "$NEW_TARGET/lfs/objects" -type f -exec stat -f '%z' {} \; |
        awk '{sum += $1} END {printf "%.0f\n", sum}'
)"

echo "Source LFS objects: $SOURCE_LFS_COUNT"
echo "Target LFS objects: $TARGET_LFS_COUNT"
echo
echo "Source LFS bytes:   $SOURCE_LFS_BYTES"
echo "Target LFS bytes:   $TARGET_LFS_BYTES"

[ "$SOURCE_LFS_COUNT" = "$TARGET_LFS_COUNT" ] ||
    fail "Source and temporary target LFS object counts differ"

[ "$SOURCE_LFS_BYTES" = "$TARGET_LFS_BYTES" ] ||
    fail "Source and temporary target LFS sizes differ"

echo
echo "=== VERIFY TEMPORARY MIRROR HEAD ==="

TARGET_SYMBOLIC_HEAD="$(
    git --git-dir="$NEW_TARGET" symbolic-ref HEAD 2>/dev/null
)" || fail "Temporary mirror has no symbolic HEAD"

TARGET_HEAD="$(
    git --git-dir="$NEW_TARGET" rev-parse "$KEEP_BRANCH"
)" || fail "Could not determine temporary mirror branch tip"

echo "Source HEAD: $SOURCE_SYMBOLIC_HEAD"
echo "Target HEAD: $TARGET_SYMBOLIC_HEAD"
echo
echo "Source tip:  $SOURCE_HEAD"
echo "Target tip:  $TARGET_HEAD"

[ "$SOURCE_SYMBOLIC_HEAD" = "$TARGET_SYMBOLIC_HEAD" ] ||
    fail "Temporary mirror HEAD differs from source"

[ "$SOURCE_HEAD" = "$TARGET_HEAD" ] ||
    fail "Temporary mirror branch tip differs from source"

echo
echo "=== VERIFY TEMPORARY MIRROR LFS ==="

git --git-dir="$NEW_TARGET" lfs fsck ||
    fail "Temporary mirror Git LFS verification failed"

echo
echo "=== VERIFY TEMPORARY MIRROR GIT ==="

git --git-dir="$NEW_TARGET" fsck --full ||
    fail "Temporary mirror Git verification failed"

echo
echo "=== INSTALL VERIFIED NETWORK MIRROR ==="

rm -rf "$OLD_TARGET"

if [ -e "$TARGET_REPO" ]; then
    mv "$TARGET_REPO" "$OLD_TARGET" ||
        fail "Could not preserve existing network repository"
fi

if ! mv "$NEW_TARGET" "$TARGET_REPO"; then
    echo "Could not install new network repository."

    if [ -d "$OLD_TARGET" ]; then
        echo "Restoring previous network repository."
        mv "$OLD_TARGET" "$TARGET_REPO"
    fi

    fail "Network repository installation failed"
fi

echo
echo "=== VERIFY INSTALLED NETWORK MIRROR ==="

INSTALLED_HEAD="$(
    git --git-dir="$TARGET_REPO" rev-parse "$KEEP_BRANCH"
)" || {
    if [ -d "$OLD_TARGET" ]; then
        rm -rf "$TARGET_REPO"
        mv "$OLD_TARGET" "$TARGET_REPO"
    fi
    fail "Installed network repository cannot resolve development branch"
}

if [ "$INSTALLED_HEAD" != "$SOURCE_HEAD" ]; then
    if [ -d "$OLD_TARGET" ]; then
        rm -rf "$TARGET_REPO"
        mv "$OLD_TARGET" "$TARGET_REPO"
    fi
    fail "Installed network branch tip differs from source"
fi

git --git-dir="$TARGET_REPO" lfs fsck ||
{
    if [ -d "$OLD_TARGET" ]; then
        rm -rf "$TARGET_REPO"
        mv "$OLD_TARGET" "$TARGET_REPO"
    fi
    fail "Installed network Git LFS verification failed"
}

git --git-dir="$TARGET_REPO" fsck --full ||
{
    if [ -d "$OLD_TARGET" ]; then
        rm -rf "$TARGET_REPO"
        mv "$OLD_TARGET" "$TARGET_REPO"
    fi
    fail "Installed network Git verification failed"
}

echo
echo "=== REMOVE SUPERSEDED NETWORK COPY ==="

rm -rf "$OLD_TARGET"

trap - EXIT

echo
echo "=== FINAL INVENTORY ==="

echo "Source tip:"
git -C "$SOURCE_REPO" rev-parse "$KEEP_BRANCH"

echo
echo "Network tip:"
git --git-dir="$TARGET_REPO" rev-parse "$KEEP_BRANCH"

echo
echo "Network HEAD:"
git --git-dir="$TARGET_REPO" symbolic-ref HEAD

echo
echo "Network repository size:"
du -sh "$TARGET_REPO"

echo
echo "Network LFS size:"
du -sh "$TARGET_REPO/lfs"

echo
echo "Network LFS object count:"
find "$TARGET_REPO/lfs/objects" -type f | wc -l

echo
echo "Historical database commits:"
git --git-dir="$TARGET_REPO" \
    log --all --oneline -- \
    Database/backups/LSTM_latest.dump

echo
echo "PASS: FULL NETWORK BACKUP COMPLETE"
echo "Full Git history and the complete local Git LFS store"
echo "have been synchronized to the network repository."
