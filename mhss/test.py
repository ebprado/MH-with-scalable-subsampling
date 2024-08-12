python3 -m build

# python3 -m twine upload dist/*   
python3 -m twine upload --repository testpypi dist/*

#Test PyPI token:
pypi-AgENdGVzdC5weXBpLm9yZwIkODBiMDY2NzUtOTMxMi00N2I3LTlmYWMtYjcyYzZmNGJmZWQ3AAIqWzMsIjBmNTI5MDRjLTkyNWEtNDBlNS1iYTM4LWFiNzYyYzQzZjU2YyJdAAAGIG5FjCCDyL7uiXv6igu9jPOq793cRTre1_6oPl64Pjc2


python3 -m pip install -i https://test.pypi.org/simple/ mhssteste


# Check permissions
# cd /Users/estevaoprado/Library/Python/3.9/lib/python/site-packages   
# ls -l

from mhssteste import *

