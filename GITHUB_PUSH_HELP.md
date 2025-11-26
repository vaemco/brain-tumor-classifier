# GitHub Push - Schnellanleitung

Du hast bereits lokal committed, aber noch nicht zu GitHub gepusht. Folge diesen Schritten:

## 1. GitHub Repository erstellen

1. Gehe zu: https://github.com/new
2. **Repository Name**: `brain-tumor-classifier` (oder dein Wunschname)
3. **Description**: "AI-powered brain tumor classification from MRI scans"
4. **Privacy**: ✅ **Private** (wie gewünscht)
5. **WICHTIG**: ❌ NICHTS auswählen bei:
   - ❌ Add a README file
   - ❌ Add .gitignore
   - ❌ Choose a license
6. Klicke "Create repository"

## 2. GitHub Remote hinzufügen

Nach dem Erstellen zeigt GitHub dir diese Befehle an. **ABER** du hast bereits einen Commit, also:

```bash
# Ersetze USERNAME mit deinem GitHub Username!
git remote add origin https://github.com/your-username/brain-tumor-classifier.git

# Verify
git remote -v
```

**Beispiel**: Wenn dein Username `your-username` ist:
```bash
git remote add origin https://github.com/your-username/brain-tumor-classifier.git
```

## 3. CSS Änderungen hinzufügen & Pushen

```bash
# Aktuelle CSS Änderungen committen
git add website/static/css/style.css
git commit -m "Fix: Add standard background-clip property for compatibility"

# Branch umbenennen zu main (falls noch master)
git branch -M main

# Zu GitHub pushen
git push -u origin main
```

## 4. Überprüfen

Gehe zu: `https://github.com/USERNAME/brain-tumor-classifier`

Du solltest jetzt sehen:
- ✅ Alle Dateien
- ✅ README.md wird angezeigt
- ✅ 2 Commits (Initial + CSS fix)
- ✅ docs/ Ordner
- ✅ LICENSE

## Häufige Fehler

### "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/USERNAME/brain-tumor-classifier.git
```

### "! [rejected] master -> main (non-fast-forward)"
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### "fatal: 'origin' does not appear to be a git repository"
→ Du hast vergessen, das Remote hinzuzufügen (Schritt 2)

### Username/Password wird gefragt
GitHub akzeptiert keine Passwörter mehr. Du brauchst:
- **Personal Access Token** ODER
- **SSH Key**

#### Personal Access Token erstellen:
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token
3. Scope: ✅ `repo` (full control)
4. Kopiere das Token
5. Verwende Token statt Passwort beim Push

#### Oder SSH verwenden:
```bash
# SSH Key generieren (wenn noch nicht vorhanden)
ssh-keygen -t ed25519 -C "deine@email.com"

# Public Key zu GitHub hinzufügen
cat ~/.ssh/id_ed25519.pub
# Kopiere und füge bei GitHub → Settings → SSH Keys ein

# Remote URL zu SSH ändern
git remote set-url origin git@github.com:USERNAME/brain-tumor-classifier.git
```

## Komplette Befehlsfolge (Copy-Paste)

```bash
# 1. CSS Änderungen committen
git add website/static/css/style.css
git commit -m "Fix: Add standard background-clip for compatibility"

# 2. Remote hinzufügen (USERNAME ERSETZEN!)
git remote add origin https://github.com/USERNAME/brain-tumor-classifier.git

# 3. Zu main umbenennen und pushen
git branch -M main
git push -u origin main
```

## Wenn alles funktioniert hat

Du solltest sehen:
```
Enumerating objects: X, done.
Counting objects: 100% (X/X), done.
Writing objects: 100% (X/X), Y KiB | Z MiB/s, done.
Total X (delta Y), reused 0 (delta 0)
To https://github.com/USERNAME/brain-tumor-classifier.git
 * [new branch]      main -> main
```

Dann gehe zu deinem Repository auf GitHub und alles sollte da sein! 🎉

## Wichtig

**Stelle sicher, dass dein GitHub Repository auf PRIVATE gesetzt ist**, da du sagtest es soll privat sein.

---

Wenn du Probleme hast, sag mir welche Fehlermeldung du bekommst!
