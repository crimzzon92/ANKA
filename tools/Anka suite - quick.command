#!/bin/bash
# Double-click: runs the "quick" suite (see "Anka suite.command" for what it does).
exec "$(dirname "$0")/Anka suite.command" quick
