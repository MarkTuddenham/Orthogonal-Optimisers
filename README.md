# Orthogonalised Optimisers

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


