#!/bin/bash

# Usage: ./search_with_wildcards.sh "pattern_with_wildcards" [directory]
# Example: ./search_with_wildcards.sh "foo*bar" ./src

PATTERN="$1"
SEARCH_DIR="${2:-.}"  # Default to current directory if not provided

if [[ -z "$PATTERN" ]]; then
  echo "Usage: $0 \"pattern_with_wildcards\" [directory]"
  exit 1
fi

# Convert shell-style wildcards to a regular expression
# * → .*
# ? → .
REGEX_PATTERN=$(echo "$PATTERN" | sed 's/\*/.*/g; s/?/./g')

# Use find to exclude files that end with 'log' (like file.log or just filelog)
# Then use grep to search inside them
find "$SEARCH_DIR" -type f ! -iname '*log' -print0 |
  xargs -0 grep -E "$REGEX_PATTERN"
