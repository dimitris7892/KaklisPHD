#!/bin/bash
# Copyright 2019 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Don't run this script standalone. Instead, run from the repository root:
# ./tools/run_tests/run_tests.py -l objc

set -ev

# Params:
# EXAMPLE_PATH - directory of the example
# SCHEME - scheme of the example, used by xcodebuild

# CocoaPods requires the terminal to be using UTF-8 encoding.
export LANG=en_US.UTF-8

cd `dirname $0`/../../..

cd $EXAMPLE_PATH/..

if [ "$SCHEME" == "watchOS-sample-WatchKit-App" ]; then
  SCHEME="watchOS-sample watchOS-sample-watchApp"
fi
../../../tools/bazel build $SCHEME
