# LSTM-LID

Source code for the paper [A reproduction of Apple’s bi-directional LSTM models for language identification in short strings](https://www.aclweb.org/anthology/2021.eacl-srw.6/).

# Installation


```bash
pip install LanguageIdentifier
```

# Example
```python
from LanguageIdentifier import predict
from LanguageIdentifier import rank

print(predict("Det her er dansk"))
print(rank("This is english")) 
```
`predict` outputs the most likely language of the sentence. `rank` returns a list of tuples containing their respective probabilities.

# Paper and bibtex
The ACL Anthology is the official location for the paper: [A reproduction of Apple’s bi-directional LSTM models for language identification in short strings](https://www.aclweb.org/anthology/2021.eacl-srw.6/).

If you use the library of paper in your work, please cite:

```bibtex
@inproceedings{toftrup-etal-2021-reproduction,
    title = "A reproduction of Apple{'}s bi-directional {LSTM} models for language identification in short strings",
    author = "Toftrup, Mads  and
      Asger S{\o}rensen, S{\o}ren  and
      Ciosici, Manuel R.  and
      Assent, Ira",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-srw.6",
    pages = "36--42",
}
```