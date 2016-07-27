#!/bin/bash

mkdir snapshots
rm snapshots/*.png

python render_one_period_movie.py 
ffmpeg -start_number 1 -i snapshots/step%04d.png movie.mp4

