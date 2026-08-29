#!/bin/sh
# Copyright 2026-present the xvec project
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

set -eu

version=4.13.1
checksum=bc13a9c57a8dd7d5196888211e5ede657cb64a3ce968608697e4f668251a8487
jar="${TMPDIR:-/tmp}/antlr-${version}-complete.jar"

if [ ! -f "$jar" ] || [ "$(sha256sum "$jar" | cut -d ' ' -f 1)" != "$checksum" ]; then
    curl -fsSL "https://www.antlr.org/download/antlr-${version}-complete.jar" -o "$jar"
fi
printf '%s  %s\n' "$checksum" "$jar" | sha256sum -c -

java -jar "$jar" -Dlanguage=Go -package antlr -no-listener SQLLexer.g4 SQLParser.g4
go run postprocess.go -- sql_parser.go
rm -f ./*.interp ./*.tokens
