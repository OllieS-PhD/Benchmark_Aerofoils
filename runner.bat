@echo off
set mod1="GUNet"
set mod2="GraphSAGE"
set mod3="MLP"
set mod4="PointNet"
set task="-t full"

python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\main.py %mod3% %task%
python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\main.py %mod4% %task%
python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\main.py %mod1% %task%
python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\main.py %mod2% %task%
