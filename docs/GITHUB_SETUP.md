# Git Setup & GitHub Push Guide

Follow these steps to push your project to GitHub.

## 1. Initialize Git Repository

```bash
cd /Users/valentinemser/dev_projects/03_data_projects/data_brain_tumor

# Initialize git
git init

# Add all files (respecting .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: Brain Tumor Classifier with Deep Learning

- ResNet18 transfer learning model
- Flask web application with Grad-CAM visualization
- Modern UI with drag & drop interface
- ~98% validation accuracy
- M2 MacBook optimized
- Comprehensive documentation"
```

## 2. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `brain-tumor-classifier` (or your choice)
3. Description: "AI-powered brain tumor classification from MRI scans with explainable AI"
4. **Private repository** (as requested)
5. **DO NOT** initialize with README, .gitignore, or license (we have them)
6. Click "Create repository"

## 3. Push to GitHub

```bash
# Add GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/brain-tumor-classifier.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

## 4. Verify Upload

Check these files are on GitHub:
- ✅ README.md (professional, with AI transparency)
- ✅ LICENSE
- ✅ .gitignore
- ✅ environment.yml
- ✅ All code files (website/, notebooks/)
- ✅ docs/PROJECT_BRIEFING.md
- ❌ data/ (should be excluded)
- ❌ Large model files except final one

## 5. Optional: Add Topics

On your GitHub repository page, add topics:
- `machine-learning`
- `deep-learning`
- `pytorch`
- `medical-imaging`
- `flask`
- `computer-vision`
- `explainable-ai`
- `transfer-learning`
- `medical-ai`

## 6. Optional: GitHub Actions (CI/CD)

Create `.github/workflows/ci.yml` for automated testing:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install pytest flake8
      - name: Lint
        run: flake8 website/
      - name: Test
        run: pytest tests/
```

## 7. Update Repository Settings

On GitHub repository page:
1. Go to Settings → General
2. Features:
   - ✅ Issues
   - ✅ Projects (optional)
   - ❌ Wiki (not needed)
3. Pull Requests:
   - ✅ Allow squash merging
4. Add description and website URL (if deployed)

## Common Git Commands

```bash
# Check status
git status

# View changes
git diff

# Add specific file
git add filename.py

# Commit changes
git commit -m "Description of changes"

# Push changes
git push

# Pull latest changes
git pull

# Create new branch
git checkout -b feature-name

# Switch branch
git checkout main

# View commit history
git log --oneline
```

## Troubleshooting

### Large files preventing push

If Git complains about large files:

```bash
# Remove from git cache
git rm --cached path/to/large/file

# Add to .gitignore
echo "path/to/large/file" >> .gitignore

# Commit and push
git commit -m "Remove large file"
git push
```

### Reset last commit (if needed)

```bash
# Undo last commit but keep changes
git reset --soft HEAD~1

# Undo last commit and discard changes
git reset --hard HEAD~1
```

## Security Notes

✅ **Included in repo**:
- Source code
- Documentation
- Configuration templates
- One final model file

❌ **NOT included in repo** (via .gitignore):
- Training data
- API keys / credentials
- Large model checkpoints (except final)
- Personal information
- Temporary files

## Next Steps

After pushing:
1. ✅ Verify repository looks good on GitHub
2. ✅ Add repository description
3. ✅ Add topics for discoverability
4. ✅ Consider adding repository social image
5. ✅ Update main README if needed
6. ✅ Share repository link on portfolio/LinkedIn

---

Ready to share your project with the world! 🚀
