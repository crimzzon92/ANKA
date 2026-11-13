#!/bin/bash
# Double-click: runs the "twodays" suite (see "Anka suite.command" for what it does).
exec "$(dirname "$0")/Anka suite.command" twodays
