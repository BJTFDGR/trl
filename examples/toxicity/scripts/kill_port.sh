#!/bin/bash

# Check if a port number was provided
if [ -z "$1" ]; then
    echo "Please provide a port number."
    exit 1
fi

PORT=$1

# Get the PID of the process listening on the given port
PID=$(lsof -t -i :$PORT)

# If a process with that port number was found, kill it
if [[ -n $PID ]]; then
    echo "Killing process with PID: $PID"
    kill -9 $PID
else
    echo "No process found listening on port $PORT"
fi
