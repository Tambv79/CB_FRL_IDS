
# Release and archival procedure

1. Publish the repository on the `main` branch after replacing the URL placeholder.
2. Create tag `v1.0.0`.
3. Draft a GitHub Release titled `CB-FRL-IDS reproducibility artifact v1.0.0`.
4. Attach the artifact ZIP and SHA-256 text file from the handoff package.
5. Paste `RELEASE_NOTES_v1.0.0.md` into the release description.
6. Publish the release.
7. For a persistent DOI, connect the public repository to Zenodo and archive the `v1.0.0`
   release; then insert the DOI into `CITATION.cff` and the manuscript before final submission.

The large artifact belongs in a Release asset, not in ordinary Git history.
