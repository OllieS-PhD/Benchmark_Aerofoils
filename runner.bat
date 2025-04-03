@echo off
set mod1="GUNet"
set mod2="GraphSAGE"
set mod3="MLP"
set mod4="PointNet"
set task="-t full"
set foils=5 20 55 150 400
set epochs=50 100 200 400

for %%e in (%epochs%) do (
for %%f in (%foils%) do (
python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\main.py %mod3% %task% -f %%f -e %%e
python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\main.py %mod4% %task% -f %%f -e %%e
python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\main.py %mod1% %task% -f %%f -e %%e
python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\main.py %mod2% %task% -f %%f -e %%e
)
)
python C:\Users\olive\Documents\Code\eXFoil\eX-Foil\validation.py