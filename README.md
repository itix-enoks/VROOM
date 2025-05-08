# Git Workflow Tutorial

This README provides a simple guide for team collaboration using Git, including how to add, commit, push, and reverse changes.

## 1. Cloning the Repository

If you haven’t cloned the repository yet:

```bash
git clone <repository-url>
cd <repository-directory>
```

## 2. Adding Changes

Track files you've modified or added:

```bash
git add <file-name>        # Add a specific file
git add .                  # Add all changes in the current directory
```

## 3. Committing Changes

Commit the added changes with a clear message:

```bash
git commit -m "Describe the changes made"
```

## 4. Pushing Changes

Push your branch to the remote repository:

```bash
git push origin <feature-branch-name>
```

## 5. Pulling Latest Changes from Main

Make sure your branch is up to date:

```bash
git checkout main
git pull origin main
git checkout <feature-branch-name>
git merge main
```

## 6. Reversing Changes

### Unstaging a file:

```bash
git reset <file-name>
```

### Undoing a local commit (but keeping the changes):

```bash
git reset --soft HEAD~1
```

### Reverting a pushed commit:

```bash
git revert <commit-hash>
```

### Discard all local changes:

**Dangerous: this will erase uncommitted changes**

```bash
git reset --hard
```