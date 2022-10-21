#!/bin/bash

python3 -m pip install -r requirements.txt

for i in 1558519716 1560149277 1561106829; do
    json=${i}.json
    if [ ! -f "$json" ]; then
        chat_downloader -o $json https://www.twitch.tv/videos/${i} > /dev/null
    fi
done

# ./main.py *.json
