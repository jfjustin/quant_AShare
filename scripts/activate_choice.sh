#!/usr/bin/env bash
# Launch the Choice LoginActivator from ITS OWN directory (required — it reads
# ServerList.json.e / assets by relative path and crashes elsewhere).
DIR="$HOME/EMQuantAPI_Python/python3/libs/mac"
cd "$DIR" || { echo "activator dir not found: $DIR"; exit 1; }
echo "Launching Choice activator from $DIR ..."
echo "A login window should appear. Log in with your Choice account."
exec ./loginactivator_mac
