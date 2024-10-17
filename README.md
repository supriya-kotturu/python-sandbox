# python-sandbox

## FAQ

### 1. How to import functions or classes from another juyter notebook?

To `import` functions from another jupyter notebook, you can use `%run` in your current file. This runs your `file_name.ipynb` and makes its fuctions available for your current file

```python
%run file_name.ipynb
```

If you want to import a function or class from `file_name.ipynb`

```python
%run linked_lists.ipynb import LinkedList, Node
```

But for some system, this might not work and you might want to follow the steps mentioned in this article: https://bobbyhadz.com/blog/import-jupyter-ipynb-file-from-another-ipynb-file

If none of the steps work and you end up getting an `mbformat module not found` error, try installing it using pip

```bash
pip install nbformat nbclient
```

**1.1. An Elegant way to import modules in jupyter notebook**

You can use `import-ipynb` before you import your file.

```python
import import_ipynb

import file_name_2              # imports or runs entire notebook
from file_name import func_1    # imports specific function from the notebook
```
