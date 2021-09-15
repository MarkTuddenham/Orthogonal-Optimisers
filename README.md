# Orthogonalised Optimisers

### Anonymised Repo
If you're looking at the anonymised repo then you can't clone/download it.
However, you can just download the relevant files individually:

```bash
mkdir -p Orthogonal-Optimiser
cd Orthogonal-Optmiser
mkdir src/orth_optim
wget https://anonymous.4open.science/api/repo/Orthogonal-Optimisers/file/setup.py
wget https://anonymous.4open.science/api/repo/Orthogonal-Optimisers/file/src/orth_optim/__init__.py -O src/orth_optim/__init__.py
```

## Install  package
```bash
git clone https://github.com/MarkTuddenham/Orthogonal-Optimiser.git
cd Orthogonal-Optimiser
pip install .
```
or 

```bash
pip install git+https://github.com/MarkTuddenham/Orthogonal-Optimiser.git#egg=orth_optim
```

## Usage
And then at the top of your main python script:

```python
from orth_optim import hook
hook()
```
Now the torch optimisers have an orthogonal option, e.g:
```python
torch.optim.SGD(model.parameters(),
                lr=1e-3,
                momentum=0.9,
                orth=True)
```

## Custom Optimisers
If you have a custom optimiser you can apply the `orthogonalise` decorator.

```python
from orth_optim import orthogonalise

@orthogonalise
class LARS(torch.optim.Optimizer):
	...
```


