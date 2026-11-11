# Publishing this repository to GitHub (VS Code)

Delete this file after the first push if you like.

## 0. One-time setup

- Install [Git](https://git-scm.com/downloads) and [VS Code](https://code.visualstudio.com/).
- In a terminal, run these once with your own name and GitHub email:

  ```bash
  git config --global user.name  "Your Name"
  git config --global user.email "you@example.com"
  ```

## 1. Put your GitHub username in the links

The repository is configured for the GitHub account `gulzara3`. If you fork or rename it, replace
`gulzara3` with your GitHub username. This fixes three files:
- the badges and links in `README.md`;
- the repository link in `CITATION.cff`;
- the commands below.

## 2a. Easiest: the Publish button

1. Open the unzipped folder: *File → Open Folder… → hmcp-mbr-cleaning-date*.
2. Open the **Source Control** panel (Ctrl+Shift+G) and click **Publish to GitHub**.
3. Sign in when asked, then choose **Public** (or Private). Keep the repository name `hmcp-mbr-cleaning-date`.
4. Select all files when asked which to include. VS Code then creates the repository, makes the first commit and pushes it.

## 2b. Alternative: the terminal

1. On GitHub, create an **empty** repository called `hmcp-mbr-cleaning-date`. Do not add a README, licence or
   .gitignore, because these files already exist.
2. In VS Code, open the terminal (Ctrl+`) in the repository folder and run:

   ```bash
   git init
   git add .
   git commit -m "Initial release: HMCP cleaning-date forecaster for MBRs"
   git branch -M main
   git remote add origin https://github.com/gulzara3/hmcp-mbr-cleaning-date.git
   git push -u origin main
   ```

## 3. After the first push

- **About box.** On the repository page, click the gear next to *About*. Add a description, such as
  "Physics-constrained prognostics: an always-valid chemical-cleaning date for membrane bioreactors". Add topics too:
  `membrane-bioreactor`, `membrane-fouling`, `prognostics`, `physics-informed-machine-learning`, `pytorch`, `water-treatment`.
- **CI.** The *Actions* tab runs the syntax and notebook checks on every push.
- **DOI.** Optionally, link the repository to [Zenodo](https://zenodo.org/account/settings/github/) and create a GitHub
  release (e.g. `v1.0.0`). Zenodo then issues a DOI that you can cite in the paper's *Data availability* statement.
