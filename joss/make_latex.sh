#!/usr/bin/env bash

echo "Downloading..."
rm -rf latex.template logo.png aas-logo.png apa.csl
wget -q https://raw.githubusercontent.com/openjournals/whedon/editor-and-reviewers-on-papers/resources/joss/latex.template
wget -q https://raw.githubusercontent.com/openjournals/whedon/editor-and-reviewers-on-papers/resources/joss/logo.png
wget -q https://raw.githubusercontent.com/openjournals/whedon/editor-and-reviewers-on-papers/resources/joss/aas-logo.png
wget -q https://raw.githubusercontent.com/openjournals/whedon/editor-and-reviewers-on-papers/resources/joss/apa.csl
echo "Done"

pandoc \
-s paper.md \
-o paper.tex \
--template latex.template \
--csl=apa.csl \
--bibliography=paper.bib \
--filter pandoc-citeproc \
-V repository="https://github.com/dfm/emcee" \
-V archive_doi="https://doi.org/10.5281/zenodo.3543502" \
-V review_issue_url="https://github.com/openjournals/joss-reviews/issues/1864" \
-V editor_url="http://juanjobazan.com" \
-V graphics="true" \
--metadata-file=metadata.yaml

# -o paper.pdf -V geometry:margin=1in \
# --pdf-engine=xelatex \
# --filter pandoc-citeproc \
# -t latex \
# -o paper.tex \
# --bibliography=paper.bib
# --from markdown+autolink_bare_uris \
# --template latex.template \
# -s paper.md
