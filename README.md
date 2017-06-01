# Spherical Harmonics for Tensorflow

Use spherical harmonics without the worries of the endless waiting for the recursive basis-definition to finish.

**Requires tensorflow & sympy to be installed**
&nbsp;
&nbsp;

### Install
```python
pip install -e git+gitlab:tensorflow/sphericalHarmonicsFlow#egg=SphericalHarmonicsFlow
```

### Import
```python
from SH import *
```


## Use
Let **pts** be a 3-dimensional dataset, where each point has some additional values (**channels**).
&nbsp;

### Initialization
```python
from SH import SH

# init placeholders
pts = tf.placeholder(tf.float32, shape=[None, 3])
center = tf.placeholder(tf.float32, shape=[3])
channels = tf.placeholder(tf.float32, shape=[None, numchannels])
```
&nbsp;

### Calculate representation graph (single Sphere)
Approximate the centered points (**pts-center**) with values (**channels**) by a spherical harmonics basis with about **numcoeffs²** basis-elements.

```python  
# express pts as spherical harmonics basis approximating channels
sh = SH(pts, center, channels, numcoeffs=4)
```
**Attention!:** *This projects all points onto a single Sphere and approximates the values on the sphere*

**Info:** All basis functions up to **numcoeffs=20** are precomputed stored in *sh_basis.pkl*. This file is loaded automatically. If you intend to use a higher number of basis functions, please consider to recompute this file before useage!    
(See *Auto-Saving Basis Functions*)
&nbsp;

### Calculate representation graph (multiple Spheres)
We define first a **radius** and a maximal number of spheres to observe **numshells**. Before approximating the channels all points are first projected onto their nearest sphere-shell.

```python  
sh = SH(pts, center, channels, numcoeffs=4, numshells=10, radius=0.1)
```
&nbsp;


### Evaluate

Assuming you have the dataset saved:
```python
sess = tf.Session()
data_positions = ...
data_values    = ...
testdata_pos   = ...
```


To get the coefficients for the least squares fit of the basis to any dataset use:
```python
coeffs = sess.run(sh.coeffs, feed_dict={pts:data_positions, center:[1,2,3], channels:data_values})
```
&nbsp;

**Use coefficients**:
Use the coefficients to estimate some other data
**(Doesn't work yet with multiple Shells)**:

```python
coeffs = sess.run(sh.approx_func, feed_dict={pts:testdata_pos, center:[1,2,3], sh.coeffs_input:coeffs})
```
&nbsp;

**Alternative Feed-variables:**
Instead of feeding **pts** and **center** you can also use ```sh.x_proj``` ( *points on a unit-sphere*; **not for multi-shell mode**) or ```sh.x_local``` ( *centered points* ).
&nbsp;
&nbsp;

### Observable Variables
Multiple Shell Mode *only*: **(M)**  

| Variable name | Function |
| ---- | ---- |
| ```sh.x_local``` | local coordinates **pts-center** |
| ```sh.x_proj``` | projection onto sphere **pts-center/\|\|pts-center\|\|** |
| ```sh.x_norm``` | norm of local coordinates **\|\|pts-center\|\|** |
| ```sh.x_theta``` | angle on unit sphere |
| ```sh.x_phi``` | angle on unit sphere |
| ```sh.numbasis``` | Size of Basis (Non-Tensor) |
| ```sh.coeffs``` | coefficients for basis-functions |
| &nbsp;&nbsp;&nbsp; .shape | [#basis, #channels] |
| &nbsp;&nbsp;&nbsp; .shape **(M)** | [#shells, #basis, #channels] |
| ```sh.x_shell_no``` | shell number of each point |
| ```sh.max_shell_no``` | theoretical maximal shell number for current dataset |
| ```sh.pts_on_shell_ind``` **(M)** | subset-tensor |
| ```sh.xs_all``` **(M)** | ```sh.x_proj``` divided into different shell-subsets |
&nbsp;
&nbsp;


### Check Basis Functions
Check what polynomials are used as the basis-functions.
*(Under the assumption that the point with coordinates (```SH.x```,```SH.y```,```SH.z```) lies on the unit-sphere)*
The polynomials are built and simplified using ```sympy```.
```python
# (optional: load precomputed basis-functions)
SH.loadBasis()

# check a basis function
SH._basis_elem_sympy(l,m)
```
&nbsp;
&nbsp;


## Settings
&nbsp;
### Increase Pseudo Inverse-Accuracy
When computing a matrix-inverse, a pseudoinverse is used. Eigenvalues with ```e < 1e-5``` are mapped to zero. Configure that by using:
```python
# add flag to SH-call
SH(..., inv_eps=1e-10 )
```
&nbsp;


### Auto-Saving Basis-Functions
The spherical harmonics basis takes some while to compute (especially for a larger basis-size). We have precomputed all functions up to **numcoeffs=20**. They are stored in *sh_basis.pkl*. This file is loaded automatically on the first call of *SH(...)*.
&nbsp;

#### Disable Auto-Loading / -Saving of the Basis
```python
# add flag to SH-call
SH(..., autosave=False )
```
&nbsp;

#### Recompute Basis
If you intend to use a higher number of basis functions, please consider to recompute this file before useage.
Open python-console, run:
```python
# drink some coffee
SH.prepareBasis( numbasis, verbose=True )
```
&nbsp;

#### Change Basis File
```python
SH.sh_basis_file = "my_filename"
```
