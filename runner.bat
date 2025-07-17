@echo off
set mod= "MLP" "PointNet" "GUNet" "GraphSAGE"
set task="-t full"
set foils= 5 20 55 150
set epochs=50 100 200 400

for %%m in (%mod%) do (
for %%f in (%foils%) do (
python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\main.py %%m -f %%f -e 400
)
)
python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\validation.py
