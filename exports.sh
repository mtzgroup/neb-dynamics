#!/bin/bash
rf=/home/alessio/Desktop/retropaths
export PYTHONPATH="$PYTHONPATH:${rf}:$(pwd)"
export SHELL=/bin/bash
export retro_folder=${rf}
export OE_LICENSE=${rf}/data/oe_license.txt
export TEMPLATE_FOLDER=${rf}/data/
export OMP_NUM_THREADS=1
