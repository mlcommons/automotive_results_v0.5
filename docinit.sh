#!/bin/bash

if [ ! -e docs ]; then
    git clone https://github.com/mlcommons/automotive_results_visualization_template.git docs
    test $? -eq 0 || exit $?
fi

python3 -m pip install -r docs/requirements.txt

if [ ! -e overrides ]; then
    cp -r docs/overrides overrides
    test $? -eq 0 || exit $?
fi

repo_owner=${AUTOMOTIVE_RESULTS_REPO_OWNER:-mlcommons}}
repo_branch=${AUTOMOTIVE_RESULTS_REPO_BRANCH:-main}}
repo_name=${AUTOMOTIVE_RESULTS_REPO_NAME:-automotive_results_${AUTOMOTIVE_RESULTS_VERSION}}
echo "repo owner: ${repo_owner}"
echo "repo branch: ${repo_branch}"
echo "repo name: ${repo_name}"
ver_num=$(cat dbversion)
let ver_num++

echo "$ver_num" > dbversion

rm -f docs/javascripts/config.js

if [ ! -e docs/javascripts/config.js ]; then
    if [ -n "${AUTOMOTIVE_RESULTS_VERSION}" ]; then
        results_version="${AUTOMOTIVE_RESULTS_VERSION}"
    else
        echo "Please export AUTOMOTIVE_RESULTS_VERSION (e.g., v0.5)"
        exit 1
    fi
    echo "const results_version=\"${results_version}\";" > docs/javascripts/config.js
    echo "var repo_owner=\"${repo_owner}\";" >> docs/javascripts/config.js
    echo "var repo_branch=\"${repo_branch}\";" >> docs/javascripts/config.js
    echo "var repo_name=\"${repo_name}\";" >> docs/javascripts/config.js
    echo "const dbVersion =\"${ver_num}\";" >> docs/javascripts/config.js
    echo "const default_category =\"${default_category}\";" >> docs/javascripts/config.js
    echo "const default_division =\"${default_division}\";" >> docs/javascripts/config.js
fi

if [ ! -e docs/thirdparty/tablesorter ]; then
    cd docs/thirdparty && git clone https://github.com/Mottie/tablesorter.git && cd -
    test $? -eq 0 || exit $?
fi

# Ensure topresults/thirdparty/tablesorter exists
if [ ! -e docs/top_results/thirdparty ]; then
    mkdir -p docs/topresults
    cp -r docs/thirdparty docs/top_results/thirdparty
    test $? -eq 0 || exit $?
fi

# Ensure compare/thirdparty/tablesorter exists
if [ ! -e docs/compare/thirdparty ]; then
    mkdir -p docs/compare
    cp -r docs/thirdparty docs/compare/thirdparty
    test $? -eq 0 || exit $?
fi

# Ensure topresults/javascripts exists
if [ ! -e docs/top_results/javascripts ]; then
    mkdir -p docs/topresults
    cp -r docs/javascripts docs/top_results/javascripts
    test $? -eq 0 || exit $?
fi

# Ensure compare/javascripts exists
if [ ! -e docs/compare/javascripts ]; then
    mkdir -p docs/compare
    cp -r docs/javascripts docs/compare/javascripts
    test $? -eq 0 || exit $?
fi

if [ ! -e process_results_table.py ]; then
    cp docs/process_results_table.py .
    test $? -eq 0 || exit $?
fi

if [ ! -e process.py ]; then
    cp docs/process.py .
    test $? -eq 0 || exit $?
fi

if [ ! -e add_results_summary.py ]; then
    cp docs/add_results_summary.py .
    test $? -eq 0 || exit $?
fi

if [ -n "${AUTOMOTIVE_RESULTS_VERSION}" ]; then
    repo_to_clone="mlperf_automotive"
else
    echo "Please export AUTOMOTIVE_RESULTS_VERSION."
    exit 1
fi

export PYTHONPATH="$repo_to_clone/tools/submission:$PYTHONPATH"

if [ ! -e "${repo_to_clone}" ]; then
    git clone https://github.com/mlcommons/${repo_to_clone} "${repo_to_clone}" --depth=1
    test $? -eq 0 || exit $?
fi


python3 process.py
test $? -eq 0 || exit $?
python3 process_results_table.py
test $? -eq 0 || exit $?

cp summary_results.json docs/javascripts/

python3 add_results_summary.py
test $? -eq 0 || exit $?
git push || (sleep $((RANDOM % 100 + 1)) && git pull --rebase && git push)

if [ -z "$(git config --global user.name)" ]; then
    git config user.name "mlcommons-bot"
    git config user.email "mlcommons-bot@users.noreply.github.com"
    echo "Git user not detected. Default mlcommons-bot username and email configured."
fi

git add '**/README.md' '**/summary.html'
git commit -m "Added results summary"
git push
