# How to Develop an Operator on Towhee Hub

## 0. Requirements

- [Towhee](https://docs.towhee.io/Getting Started/install/)
- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) *>= 1.8.2*

> If you want to upload large files, install [Git LFS](https://git-lfs.github.com/) as well.

## 1.  Create an Account and Login 

Towhee Hub is not currently open for registration, so if you'd like to create an account, please contact us on the [Slack](https://slack.towhee.io) channel.  We would love to hear about your contributions and welcome your involvement with Towhee.

Once you have an account, you can [sign in](https://towhee.io/user/login) with your username and password.

![img](https://github.com/towhee-io/data/blob/main/image/docs/login.png?raw=true)

## 2. New Operator

After signing in, you can click on your account avatar and select **New Operator** to create an operator:

![img](https://github.com/towhee-io/data/blob/main/image/docs/new_operator.png?raw=true)

Next, you can specify the **Operator Name**, set **Public**, and run **Create Operator**:

![img](https://github.com/towhee-io/data/blob/main/image/docs/create_operator.png?raw=true)

Then you will get the initialized operator:

> Towhee Hub's Operator is similar to GitHub's Repository, and the commands for clone and push are the same as the `git` command.

![img](https://github.com/towhee-io/data/blob/main/image/docs/my_operator.png?raw=true)

## 3. Develop the Operator

### Clone the Operator Repo

On the operator repo page, click **Files and versions,** and **copy** the link to clone repository:

![img](https://github.com/towhee-io/data/blob/main/image/docs/clone_operator.png?raw=true)

Run the `git` command to clone the repository with the copied link:

```Bash
$ git clone https://towhee.io/towhee/my-operator.git
```

### Develop the Code

In general, the operator repository contains the following files:

- **your_operator_name.py**, the main Python Operator file, with the same name as your operator. The Operator class is defined, and it requires `__init__` and `__call__` methods.
- **__init__.py**, the main file when importing operators. It is supposed to define a method for the operator, which will call the Operator class in the Python Operator file.
- **requirements.txt**, maintains project-related dependencies. Towhee automatically checks and installs dependencies when the operator is run for the first time.
- **README.md**, the README for the operator, usually includes sections for Description, Code Examples, Factory Constructors and Interface, etc.

The following is an example of how to calculate the inner product of two vectors in [towhee/my-operator](https://towhee.io/towhee/my-operator). First, go into the repo dictionary and create new files.

```Bash
$ cd my-operator
$ touch my_operator.py __init__.py requirements.txt
```

Then develop in those files:

**my_operator.py**

This Python file defines the `MyOperator` class, where the `__init__` method is used to initialize the Operator, and the `__call__` method is used to call the Operator.

> PyOperator is the superclass of the Python method class; if your operator is used to process models, use NNOperator, such as [image-embedding/timm](https://towhee.io/image-embedding/timm).

```Python
import numpy as np
from towhee.operator import PyOperator

class MyOperator(PyOperator):
    def __init__(self):
        super().__init__()

    def __call__(self, x, y):
        return np.inner(x, y)
```

**__init__.py**

This Python file defines the `my_operator` method (the method name is consistent with the operator name), which is used to return the initialized MyOperator class.

```Python
from .my_operator import MyOperator

def my_operator(*args, **kwargs):
    return MyOperator(*args, **kwargs)
```

**requirements.txt**

This file specifies the python package needed for each line, and `numpy` is used for this operator.

```Bash
numpy
```

**README.md**

Finally, update the [README](https://towhee.io/towhee/my-operator/src/branch/main/README.md) so that other users can run this Operator according to the tutorial.

## 4. Push the Operator to Towhee Hub

Push the developed files to Towhee Hub:

> - You need to enter your username and password for the first time.
> - If there are some large files, check that the file type is in the **.gitattributes** file and push as usual.

```Bash
$ git add __init__.py my_operator.py requirements.txt README.md
$ git commit -m "Add my-operator"
$ git push
```

## 5. Run the Operator

Then you can run the operator with `ops` or in a pipeline with `pipe`, and the **towhee/my_operator** repository will be downloaded to the **~/.towhee/opertors** directory.

### Run with ops

```Python
from towhee import ops

op = ops.towhee.my_operator()
res = op([1, 2, 3], [4, 5, 6])
```

### Run a Pipeline

```Python
from towhee import ops, pipe

p = (pipe.input('x', 'y')
         .map(('x', 'y'), 'res', ops.towhee.my_operator())
         .output('res')
     )
res = p([1, 2, 3], [4, 5, 6]).get()
```
