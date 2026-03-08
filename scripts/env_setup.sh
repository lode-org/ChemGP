#!/usr/bin/env bash

export SETUPTOOLS_SCM_PRETEND_VERSION

SETUPTOOLS_SCM_PRETEND_VERSION="99.0.dev0+chemgp_$(git rev-parse HEAD)"

export GITROOT

GITROOT=$(git rev-parse --show-toplevel)
