@echo off
set mod1=
set mod2=
set mod3="MLP"
set mod="PointNet" "GUNet" "GraphSAGE"
set task="-t full"
set foils= 55 150 400
set epochs=50 100 200 400

for %%m in (%mod%) do (
python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\main.py %%m %task% -f 400
)
python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\validation.py