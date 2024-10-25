#!/bin/bash

# script to refresh and autogenerate docs and host them locally

rm -rf ./docs/source/*
sphinx-apidoc -e -o docs/source . conf.py
rm -rf _build/ && make html & python3 -m http.server -d ./_build/html