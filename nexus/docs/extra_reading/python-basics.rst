.. _python-basics:

Basic Python Constructs
=======================

Basic Python data types (``int``, ``float``, ``str``, ``tuple``,
``list``, ``dict``) and programming constructs
(``if`` statements, ``for`` loops, functions w/ keyword arguments) are
briefly overviewed below. All examples can be executed interactively in
Python. To do this, type “``python``” at the command line and paste any
of the shaded text below at the “``>>>``” prompt. For more information
about effective use of Python, consult the detailed online
documentation: https://docs.python.org/3/.

Intrinsic types: ``int, float, str``
------------------------------------

.. code-block:: python

    # Comments start with a # symbol and continue to the end of the line
    >>> i = 5                     # integer
    >>> f = 3.6                   # float
    >>> s = 'quantum/monte/carlo' # string
    >>> n = None                  # represents "nothing"

    >>> f += 1.4 # add-assign (-, *, / also)
    >>> f
    5.0

    >>> 2**3 # raise to a power
    8

    >>> str(i) # int to string
    '5'

    >>> s + '/simulations' # joining strings
    'quantum/monte/carlo/simulations'

    >>> 'i = {0}'.format(i) # format string
    'i = 5'

    >>> f"i = {i}" # Alternate type of format string
    'i = 5'

Container types: ``tuple, list, array, dict, obj``
--------------------------------------------------

.. code-block:: python

    >>> import numpy as np       # Widely accepted to abbreviate numpy to `np`
    >>> from nexus import obj    # Get obj from Nexus

    >>> t = ('A', 42, 56, 123.0) # tuple

    >>> l = ['B', 3.14, 196]     # list

    >>> a = np.array([1, 2, 3])  # array

    >>> d = {'a':5, 'b':6}       # dict

    >>> o = obj(a=5, b=6)        # obj

    # Printing
    >>> print(t)
    ('A', 42, 56, 123.0)
    >>> print(l)
    ['B', 3.14, 196]
    >>> print(a)
    [1 2 3]
    >>> print(d)
    {'a': 5, 'b': 6}
    >>> print(o)
      a               = 5
      b               = 6

    >>> len(t), len(l), len(a), len(d), len(o) # Number of elements
    (4, 3, 3, 2, 2)

    >>> t[0], l[0], a[0], d['a'], o.a # Element access
    ('A', 'B', 1, 5, 5)

    >>> s = np.array([0,1,2,3,4])  # Slices: works for tuple, list, array
    >>> s[:]     # All elements
    array([0, 1, 2, 3, 4])
    >>> s[2:]    # Slice 2 to end
    array([2, 3, 4])
    >>> s[:2]    # Slice start until 2, not including 2
    array([0, 1])
    >>> s[1:4]   # Slice 1 until 4, not including 4
    array([1, 2, 3])
    >>> s[::2]   # Slice beginning to end, in steps of 2
    array([0, 2, 4])

    # List operations
    >>> l = ['B', 3.14, 196]
    >>> l2 = list(l) # Make independent copy
    >>> l2[0] = 1    # Set first element of list to 1
    >>> l2
    [1, 3.14, 196]
    >>> l
    ['B', 3.14, 196] # Not altered

    >>> l.append(4) # Add new element to the end of the list
    >>> l
    ['B', 3.14, 196, 4]

    >>> l + [5, 6, 7] # List Addition
    ['B', 3.14, 196, 4, 5, 6, 7]

    >>> [0, 1] * 3 # List Multiplication
    [0, 1, 0, 1, 0, 1]

    # Array Operations
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([5, 6, 7])
    >>> a2 = a.copy() # Make independent copy

    >>> a + b # Addition of two arrays
    array([ 6,  8, 10])
    >>> a + 3 # Add a scalar to an array. Adds to all elements!
    array([4, 5, 6])
    >>> a * b # Multiply two arrays of the same shape
    array([ 5, 12, 21])
    >>> a * 3 # Multiply an array by a scalar. Multiplies all elements!
    array([3, 6, 9])

    # dict/obj operations
    >>> d = {'a':5, 'b':6}
    >>> d.copy() # Independent copy
    {'a': 5, 'b': 6}
    >>> d2 = d.copy()
    >>> d2['a'] = 2 # Assign element
    >>> d2
    {'a': 2, 'b': 6}
    >>> d # Not altered.
    {'a': 5, 'b': 6}

    >>> d['c'] = 7 # Add element
    >>> d
    {'a': 5, 'b': 6, 'c': 7}

    >>> d.keys() # Get dictionary keys
    dict_keys(['a', 'b', 'c'])
    >>> d.values() # Get dictionary values
    dict_values([5, 6, 7])

    # obj-specific operations
    >>> o = obj(a=5, b=6)
    >>> o.c = 7 # Add/assign element
    >>> print(o)
    >>> print(o)
      a               = 5
      b               = 6
      c               = 7

    >>> o = o.set(d=8, e=9) # Add/assign multiple elements at a time
    >>> print(o)
      a               = 5
      b               = 6
      c               = 7
      d               = 8
      e               = 9


An important feature of Python to be aware of is that assignment is most
often by reference, *i.e.* new values are not always created. This point
is illustrated below with an ``obj`` instance, but it also holds for
``list``, ``array``, ``dict``, and others.

.. code-block:: python

    >>> o = obj(a=5, b=6)
    >>> p = o
    >>> p.a = 7
    >>> print(o)
      a               = 7
      b               = 6

    >>> q = o.copy()
    >>> q.a = 9
    >>> print(o)
      a               = 7
      b               = 6

Here ``p`` is just another name for ``o``, while ``q`` is a fully
independent copy of it.

Conditional Statements: ``if/elif/else``
----------------------------------------

.. code-block:: python

    >>> a = 5
    >>> if a is None: # Use `is` for comparison to `None`
    ...     print('a is None')
    ... elif a == 4:
    ...     print('a is 4')
    ... elif 2 < a <= 6:
    ...     print('a is in the range (2,6]')
    ... elif a < -1 or a > 26:
    ...     print('a is not in the range [-1,26]')
    ... elif a != 10:
    ...     print('a is not 10')
    ... else:
    ...     print('a is 10')
    ... #end if
    a is in the range (2,6]

The “``#end if``” is not part of Python syntax, but you will see text
like this throughout Nexus for clear encapsulation. It is not mandatory
to include this for new code, however if you think it improves
readability then feel free to use it.

Iteration: ``for``
------------------

Iteration in Python can be done in many ways. Consider the following basic
example of iterating over the indices of a list:

.. code-block:: python

    >>> l = [1, 2, 3]
    >>> m = [4, 5, 6]
    >>> s = 0
    >>> for i in range(len(l)): # Yields i = 0, 1, 2
    ...     s += l[i] + m[i]
    ... #end for
    >>> print(s)
    21

This is perhaps the simplest way to iterate over the two lists at the same time.
While this may appear to be the best way, consider some alternative methods.
The first alternative uses ``enumerate`` to provide both the indices *and* items
in the first list ``l``.

.. code-block:: python

    >>> l = [1, 2, 3]
    >>> m = [4, 5, 6]
    >>> s = 0
    >>> for i, l_val in enumerate(l):
    ...     s += l_val + m[i]
    ... #end for
    >>> print(s)
    21

This is slightly more complex, and eliminates the indexing into ``l`` that was done
previously.
However, there is still a better way to iterate over these specific lists, which we
will accomplish with ``zip``.

.. code-block:: python

    >>> l = [1, 2, 3]
    >>> m = [4, 5, 6]
    >>> s = 0
    >>> for l_val, m_val in zip(l, m):
    ...     s += l_val + m_val
    ... #end for
    >>> print(s)
    21

Here, we have "zipped" together the two lists and are iterating over them at the
same time.
This means we have no indexing in the actual logic of the loop, which can reduce
the performance cost of indexing into a list.

An additional benefit from ``zip`` over iterating over list indices is that, if
you have two lists where one is longer than the other, ``zip`` will only iterate
until the smaller of the two lists is exhausted.

.. code-block:: python

    >>> l = [1, 2, 3, 4, 5]
    >>> m = [6, 7, 8]
    >>> s = 0
    >>> for l_val, m_val in zip(l, m): # 1, 2, 3 and 6, 7, 8
    ...     s += l_val + m_val
    ... #end for
    >>> print(s)
    27

Iterating over a ``dict`` is a common operation in Python, and can often be a
stumbling point for new Python users.
Looping over a bare dict iterates over its keys, not its values!

.. code-block:: python

    >>> d = {'a': 5, 'b': 6}
    >>> for key in d:
    ...     print(key)
    ...
    a
    b

To iterate over a dictionary's values, you must use ``dict.keys()``.

.. code-block:: python

    >>> for value in d.values():
    ...     print(value)
    ...
    5
    6

If you want to iterate over both the keys *and* the values, use ``dict.items()``.

.. code-block:: python

    >>> for key, value in d.items():
    ...     print(key, value)
    ...
    a 5
    b 6

Code internal to Nexus often uses the ``obj`` container class instead of ``dict``.
This class has very similar properties to ``dict``s and can often be used in the
same places as a ``dict``.

.. warning::
    Despite similarities between ``obj`` and ``dict``, ``obj``'s default iterator
    yields values, not keys!

.. code-block:: python

    >>> o = obj(a=5, b=6)
    >>> s = 0
    >>> for v in o: # Loop over obj values
    ...     s += v
    ... #end for
    >>> print(s)
    11

    >>> d = {'a': 5, 'b': 4}
    >>> for n, v in o.items(): # Loop over key/value pairs in obj
    ...     d[n] += v
    ... #end for
    >>> print(d)
    {'a': 10, 'b': 10}

Functions: ``def``, argument syntax
-----------------------------------

.. code:: python

    >>> def f(a, b, c=5): # Basic function, c has a default value
    ...     print(a, b, c)
    ... #end def f

    >>> f(1, b=2)
    1 2 5

    >>> def f(*args,**kwargs): # General function, returns nothing
    ...     print(args)        #     args: tuple of positional arguments
    ...     print(kwargs)      #   kwargs: dict of keyword arguments
    ... #end def f

    >>> f('s', (1, 2), a=3, b='t') # 2 positional, 2 keyword args
    ('s', (1, 2))
    {'a': 3, 'b': 't'}

    >>> l = [0, 1, 2]
    >>> f(*l, a=6) # pos. args from list, 1 kw. arg, prints:
    (0, 1, 2)
    {'a': 6}

    >>> o = obj(a=5, b=6)
    >>> f(*l, **o) # pos./kw. args from list/obj
    (0, 1, 2)
    {'a': 5, 'b': 6}

    >>> f( # Indented kw. args
    ...     blocks   = 200,
    ...     steps    = 10,
    ...     timestep = 0.01,
    ...     )
    ()
    {'steps': 10, 'blocks': 200, 'timestep': 0.01}

    >>> o = obj( # obj w/ indented kw. args
    ...     blocks   = 100,
    ...     steps    = 5,
    ...     timestep = 0.02,
    ...     )
    >>> f(**o) # kw. args from obj
    ()
    {'timestep': 0.02, 'blocks': 100, 'steps': 5}


Something you may see in some parts of Nexus are function argument type hints.
These are a way to communicate to users how a function should be used, but do
not actually enforce that the supplied arguments are of the specified type.

They should be considered a suggestion, not a type check.

.. code-block:: python

    >>> def f(a: int, b: float, c: str | None = None):
    ...     if c is not None:
    ...         print(a, b, c)
    ...     else:
    ...         print(a, b)

These type hints indicate that ``a`` is *supposed* to be an integer, ``b`` is
*supposed* to be a float, and ``c`` is **optionally** *supposed* to be a string.
