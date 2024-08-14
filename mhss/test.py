python3 -m build

python3 -m twine upload dist/*   
# python3 -m twine upload --repository testpypi dist/*

#Test PyPI token:
pypi-AgENdGVzdC5weXBpLm9yZwIkODBiMDY2NzUtOTMxMi00N2I3LTlmYWMtYjcyYzZmNGJmZWQ3AAIqWzMsIjBmNTI5MDRjLTkyNWEtNDBlNS1iYTM4LWFiNzYyYzQzZjU2YyJdAAAGIG5FjCCDyL7uiXv6igu9jPOq793cRTre1_6oPl64Pjc2

#PyPI token:
pypi-AgEIcHlwaS5vcmcCJDczNWU3NGJiLWY4ZDgtNGUwYS05MzY4LWZmOTA4YTQ5NGUxZQACKlszLCIyNGFjY2E4Mi00YTc1LTQxZjMtYTMyYy00ZGEzZWExODZiZjUiXQAABiBzTk9AZV1z-HsoPEXGS477QB0iTNWDFjL7JxbOU-vJjQ

python3 -m pip install -i https://test.pypi.org/simple/ PyMHSS

python3 -m pip install PyMHSS


# Check permissions
# cd /Users/estevaoprado/Library/Python/3.9/lib/python/site-packages   
# ls -l

from mhssteste import *

