*** Redistribution and use in source, with or without modifications, are permitted for academic purposes, provided that the proper acknowledgements are done.

##Example code
```
from LanguageIdentifier import predict
from LanguageIdentifier import rank

print(predict("Det her er dansk"))
print(rank("This is english")) 
```
predict outputs the most likely language of the sentence.\
rank returns a list of tuples containing their respective probabilities.
```
da
[('ca', 2.175456259578823e-09), ('cs', 1.421048256844415e-08), ('da', 1.600260475242976e-06), ('de', 2.074262738460675e-06), ('en', 0.9998534917831421), ('es', 2.0248389631660757e-08), ('et', 9.649543386558435e-08), ('fi', 1.3886580063626752e-06), ('fr', 2.1743964850884367e-07), ('hr', 6.692872034363972e-08), ('hu', 0.00012513755063991994), ('it', 1.0744290079856e-07), ('lt', 9.55935547608533e-07), ('nl', 1.4533302419295069e-06), ('no', 6.45266709398129e-06), ('pl', 3.1909507924865466e-07), ('pt', 1.301345520232644e-07), ('ro', 6.668889795946598e-08), ('sv', 6.2374842855206225e-06), ('tr', 4.649361429187593e-08)]```