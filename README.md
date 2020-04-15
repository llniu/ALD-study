
### AFLD project
- roughly 500 AFLD patients (to varying degrees) and 200 controls

- Link to repository: [github.com/llniu/ALD-study](https://github.com/llniu/ALD-study)

### PRIDE Archives from Lili's paper on NAFLD:

- [PXD011839](https://www.ebi.ac.uk/pride/archive/projects/PXD011839) (human)
- [PXD012056](https://www.ebi.ac.uk/pride/archive/projects/PXD012056) (mice)

```
from bioservices.pride import PRIDE
client = PRIDE(verbose=True, cache=False)
```