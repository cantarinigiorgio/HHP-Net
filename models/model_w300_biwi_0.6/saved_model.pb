??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8̥
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	2?*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?x*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?x*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:x*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xZ*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:xZ*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:Z*
dtype0
p

yaw/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_name
yaw/kernel
i
yaw/kernel/Read/ReadVariableOpReadVariableOp
yaw/kernel*
_output_shapes

:Z*
dtype0
h
yaw/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
yaw/bias
a
yaw/bias/Read/ReadVariableOpReadVariableOpyaw/bias*
_output_shapes
:*
dtype0
t
pitch/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_namepitch/kernel
m
 pitch/kernel/Read/ReadVariableOpReadVariableOppitch/kernel*
_output_shapes

:Z*
dtype0
l

pitch/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
pitch/bias
e
pitch/bias/Read/ReadVariableOpReadVariableOp
pitch/bias*
_output_shapes
:*
dtype0
r
roll/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameroll/kernel
k
roll/kernel/Read/ReadVariableOpReadVariableOproll/kernel*
_output_shapes

:Z*
dtype0
j
	roll/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	roll/bias
c
roll/bias/Read/ReadVariableOpReadVariableOp	roll/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?L
value?LB?L B?L
?
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-3
layer-16
layer-17
layer_with_weights-4
layer-18
layer-19
layer_with_weights-5
layer-20
layer-21
layer_with_weights-6
layer-22
layer_with_weights-7
layer-23
layer_with_weights-8
layer-24
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
R
regularization_losses
 trainable_variables
!	variables
"	keras_api
R
#regularization_losses
$trainable_variables
%	variables
&	keras_api
R
'regularization_losses
(trainable_variables
)	variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
^

1kernel
2regularization_losses
3trainable_variables
4	variables
5	keras_api
h

6kernel
7bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
R
<regularization_losses
=trainable_variables
>	variables
?	keras_api

@	keras_api
R
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
R
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
R
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
R
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
R
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
R
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
R
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
h

]kernel
^bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
R
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
h

gkernel
hbias
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
R
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
h

qkernel
rbias
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
R
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
i

{kernel
|bias
}regularization_losses
~trainable_variables
	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
?
+0
,1
12
63
74
]5
^6
g7
h8
q9
r10
{11
|12
?13
?14
?15
?16
?
+0
,1
12
63
74
]5
^6
g7
h8
q9
r10
{11
|12
?13
?14
?15
?16
?
?metrics
?non_trainable_variables
regularization_losses
?layer_metrics
 ?layer_regularization_losses
trainable_variables
	variables
?layers
 
 
 
 
?
?metrics
?non_trainable_variables
regularization_losses
?layer_metrics
 ?layer_regularization_losses
 trainable_variables
!	variables
?layers
 
 
 
?
?metrics
?non_trainable_variables
#regularization_losses
?layer_metrics
 ?layer_regularization_losses
$trainable_variables
%	variables
?layers
 
 
 
?
?metrics
?non_trainable_variables
'regularization_losses
?layer_metrics
 ?layer_regularization_losses
(trainable_variables
)	variables
?layers
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
?
?metrics
?non_trainable_variables
-regularization_losses
?layer_metrics
 ?layer_regularization_losses
.trainable_variables
/	variables
?layers
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

10

10
?
?metrics
?non_trainable_variables
2regularization_losses
?layer_metrics
 ?layer_regularization_losses
3trainable_variables
4	variables
?layers
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

60
71
?
?metrics
?non_trainable_variables
8regularization_losses
?layer_metrics
 ?layer_regularization_losses
9trainable_variables
:	variables
?layers
 
 
 
?
?metrics
?non_trainable_variables
<regularization_losses
?layer_metrics
 ?layer_regularization_losses
=trainable_variables
>	variables
?layers
 
 
 
 
?
?metrics
?non_trainable_variables
Aregularization_losses
?layer_metrics
 ?layer_regularization_losses
Btrainable_variables
C	variables
?layers
 
 
 
?
?metrics
?non_trainable_variables
Eregularization_losses
?layer_metrics
 ?layer_regularization_losses
Ftrainable_variables
G	variables
?layers
 
 
 
?
?metrics
?non_trainable_variables
Iregularization_losses
?layer_metrics
 ?layer_regularization_losses
Jtrainable_variables
K	variables
?layers
 
 
 
?
?metrics
?non_trainable_variables
Mregularization_losses
?layer_metrics
 ?layer_regularization_losses
Ntrainable_variables
O	variables
?layers
 
 
 
?
?metrics
?non_trainable_variables
Qregularization_losses
?layer_metrics
 ?layer_regularization_losses
Rtrainable_variables
S	variables
?layers
 
 
 
?
?metrics
?non_trainable_variables
Uregularization_losses
?layer_metrics
 ?layer_regularization_losses
Vtrainable_variables
W	variables
?layers
 
 
 
?
?metrics
?non_trainable_variables
Yregularization_losses
?layer_metrics
 ?layer_regularization_losses
Ztrainable_variables
[	variables
?layers
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

]0
^1
?
?metrics
?non_trainable_variables
_regularization_losses
?layer_metrics
 ?layer_regularization_losses
`trainable_variables
a	variables
?layers
 
 
 
?
?metrics
?non_trainable_variables
cregularization_losses
?layer_metrics
 ?layer_regularization_losses
dtrainable_variables
e	variables
?layers
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

g0
h1

g0
h1
?
?metrics
?non_trainable_variables
iregularization_losses
?layer_metrics
 ?layer_regularization_losses
jtrainable_variables
k	variables
?layers
 
 
 
?
?metrics
?non_trainable_variables
mregularization_losses
?layer_metrics
 ?layer_regularization_losses
ntrainable_variables
o	variables
?layers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

q0
r1
?
?metrics
?non_trainable_variables
sregularization_losses
?layer_metrics
 ?layer_regularization_losses
ttrainable_variables
u	variables
?layers
 
 
 
?
?metrics
?non_trainable_variables
wregularization_losses
?layer_metrics
 ?layer_regularization_losses
xtrainable_variables
y	variables
?layers
VT
VARIABLE_VALUE
yaw/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEyaw/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

{0
|1

{0
|1
?
?metrics
?non_trainable_variables
}regularization_losses
?layer_metrics
 ?layer_regularization_losses
~trainable_variables
	variables
?layers
XV
VARIABLE_VALUEpitch/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
pitch/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?metrics
?non_trainable_variables
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?trainable_variables
?	variables
?layers
WU
VARIABLE_VALUEroll/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	roll/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?metrics
?non_trainable_variables
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?trainable_variables
?	variables
?layers
 
 
 
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d/kernelconv1d/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasroll/kernel	roll/biaspitch/kernel
pitch/bias
yaw/kernelyaw/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_71853831
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpyaw/kernel/Read/ReadVariableOpyaw/bias/Read/ReadVariableOp pitch/kernel/Read/ReadVariableOppitch/bias/Read/ReadVariableOproll/kernel/Read/ReadVariableOproll/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_71854588
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv1d_2/kernelconv1d_1/kernelconv1d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias
yaw/kernelyaw/biaspitch/kernel
pitch/biasroll/kernel	roll/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_71854649ب
?
g
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_71853199

inputs
identityh
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:?????????*
alpha%???=2
	LeakyReluo
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_pitch_layer_call_and_return_conditional_losses_71853462

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?	
?
C__inference_pitch_layer_call_and_return_conditional_losses_71854484

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
|
'__inference_roll_layer_call_fn_71854512

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_roll_layer_call_and_return_conditional_losses_718534362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Z::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
t
H__inference_multiply_1_layer_call_and_return_conditional_losses_71854349
inputs_0
inputs_1
identityW
mulMulinputs_0inputs_1*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
(__inference_model_layer_call_fn_71853679
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_718536382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
p
F__inference_multiply_layer_call_and_return_conditional_losses_71853269

inputs
inputs_1
identityU
mulMulinputsinputs_1*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_71854082

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_718536382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_71854201

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slicen
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_71853213

inputs
identityh
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:?????????*
alpha%???=2
	LeakyReluo
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_71853255

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_2_layer_call_and_return_conditional_losses_71853397

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xZ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_71854193

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slicen
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_layer_call_and_return_conditional_losses_71854378

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_71854304

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv1d_layer_call_and_return_conditional_losses_71854226

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_dense_1_layer_call_fn_71854416

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_718533582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
r
H__inference_multiply_1_layer_call_and_return_conditional_losses_71853284

inputs
inputs_1
identityU
mulMulinputsinputs_1*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

b
F__inference_lambda_2_layer_call_and_return_conditional_losses_71854163

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_sliceS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *?W?2
sub/yo
subSubstrided_slice:output:0sub/y:output:0*
T0*+
_output_shapes
:?????????2
subS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??G@2
mul/x`
mulMulmul/x:output:0sub:z:0*
T0*+
_output_shapes
:?????????2
mul_
IdentityIdentitymul:z:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_lambda_layer_call_and_return_conditional_losses_71853095

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slicen
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_pitch_layer_call_fn_71854493

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_pitch_layer_call_and_return_conditional_losses_718534622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Z::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
G
+__inference_lambda_2_layer_call_fn_71854185

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_718530662
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_71854125

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_718537452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
s
I__inference_concatenate_layer_call_and_return_conditional_losses_71853300

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????22
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?H
?
$__inference__traced_restore_71854649
file_prefix"
assignvariableop_conv1d_kernel"
assignvariableop_1_conv1d_bias&
"assignvariableop_2_conv1d_2_kernel&
"assignvariableop_3_conv1d_1_kernel$
 assignvariableop_4_conv1d_1_bias#
assignvariableop_5_dense_kernel!
assignvariableop_6_dense_bias%
!assignvariableop_7_dense_1_kernel#
assignvariableop_8_dense_1_bias%
!assignvariableop_9_dense_2_kernel$
 assignvariableop_10_dense_2_bias"
assignvariableop_11_yaw_kernel 
assignvariableop_12_yaw_bias$
 assignvariableop_13_pitch_kernel"
assignvariableop_14_pitch_bias#
assignvariableop_15_roll_kernel!
assignvariableop_16_roll_bias
identity_18??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv1d_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv1d_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_yaw_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_yaw_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_pitch_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_pitch_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_roll_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_roll_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_169
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_17?
Identity_18IdentityIdentity_17:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_18"#
identity_18Identity_18:output:0*Y
_input_shapesH
F: :::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

b
F__inference_lambda_2_layer_call_and_return_conditional_losses_71854175

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_sliceS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *?W?2
sub/yo
subSubstrided_slice:output:0sub/y:output:0*
T0*+
_output_shapes
:?????????2
subS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??G@2
mul/x`
mulMulmul/x:output:0sub:z:0*
T0*+
_output_shapes
:?????????2
mul_
IdentityIdentitymul:z:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
r
F__inference_multiply_layer_call_and_return_conditional_losses_71854337
inputs_0
inputs_1
identityW
mulMulinputs_0inputs_1*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
F__inference_conv1d_2_layer_call_and_return_conditional_losses_71854247

inputs/
+conv1d_expanddims_1_readvariableop_resource
identity??"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
IdentityIdentityconv1d/Squeeze:output:0#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_1_layer_call_fn_71854298

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_718531992
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
u
I__inference_concatenate_layer_call_and_return_conditional_losses_71854362
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????22
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
g
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_71854450

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????Z*
alpha%???=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????Z:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
?
F__inference_conv1d_1_layer_call_and_return_conditional_losses_71853123

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
!__inference__traced_save_71854588
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop)
%savev2_yaw_kernel_read_readvariableop'
#savev2_yaw_bias_read_readvariableop+
'savev2_pitch_kernel_read_readvariableop)
%savev2_pitch_bias_read_readvariableop*
&savev2_roll_kernel_read_readvariableop(
$savev2_roll_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop%savev2_yaw_kernel_read_readvariableop#savev2_yaw_bias_read_readvariableop'savev2_pitch_kernel_read_readvariableop%savev2_pitch_bias_read_readvariableop&savev2_roll_kernel_read_readvariableop$savev2_roll_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 * 
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::	2?:?:	?x:x:xZ:Z:Z::Z::Z:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	2?:!

_output_shapes	
:?:%!

_output_shapes
:	?x: 	

_output_shapes
:x:$
 

_output_shapes

:xZ: 

_output_shapes
:Z:$ 

_output_shapes

:Z: 

_output_shapes
::$ 

_output_shapes

:Z: 

_output_shapes
::$ 

_output_shapes

:Z: 

_output_shapes
::

_output_shapes
: 
?
F
*__inference_flatten_layer_call_fn_71854309

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_718532552
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_lambda_layer_call_fn_71854146

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_718530872
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_roll_layer_call_and_return_conditional_losses_71853436

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?`
?
C__inference_model_layer_call_and_return_conditional_losses_71853507
input_1
conv1d_1_71853134
conv1d_1_71853136
conv1d_2_71853160
conv1d_71853189
conv1d_71853191
dense_71853330
dense_71853332
dense_1_71853369
dense_1_71853371
dense_2_71853408
dense_2_71853410
roll_71853447
roll_71853449
pitch_71853473
pitch_71853475
yaw_71853499
yaw_71853501
identity

identity_1

identity_2??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?pitch/StatefulPartitionedCall?roll/StatefulPartitionedCall?yaw/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_718530212
lambda_1/PartitionedCall?
lambda_2/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_718530542
lambda_2/PartitionedCall?
lambda/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_718530872
lambda/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_1_71853134conv1d_1_71853136*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_718531232"
 conv1d_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv1d_2_71853160*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_718531512"
 conv1d_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv1d_71853189conv1d_71853191*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_718531782 
conv1d/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_718531992
leaky_re_lu_1/PartitionedCall?
tf.math.sigmoid/SigmoidSigmoid)conv1d_2/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2
tf.math.sigmoid/Sigmoid?
leaky_re_lu/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_718532132
leaky_re_lu/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_718532272
flatten_1/PartitionedCall?
flatten_2/PartitionedCallPartitionedCalltf.math.sigmoid/Sigmoid:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_718532412
flatten_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_718532552
flatten/PartitionedCall?
multiply/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_multiply_layer_call_and_return_conditional_losses_718532692
multiply/PartitionedCall?
multiply_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_multiply_1_layer_call_and_return_conditional_losses_718532842
multiply_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0#multiply_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_718533002
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_71853330dense_71853332*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_718533192
dense/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_718533402
leaky_re_lu_2/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0dense_1_71853369dense_1_71853371*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_718533582!
dense_1/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_718533792
leaky_re_lu_3/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0dense_2_71853408dense_2_71853410*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_718533972!
dense_2/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_718534182
leaky_re_lu_4/PartitionedCall?
roll/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0roll_71853447roll_71853449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_roll_layer_call_and_return_conditional_losses_718534362
roll/StatefulPartitionedCall?
pitch/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0pitch_71853473pitch_71853475*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_pitch_layer_call_and_return_conditional_losses_718534622
pitch/StatefulPartitionedCall?
yaw/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0yaw_71853499yaw_71853501*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_yaw_layer_call_and_return_conditional_losses_718534882
yaw/StatefulPartitionedCall?
IdentityIdentity$yaw/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^pitch/StatefulPartitionedCall^roll/StatefulPartitionedCall^yaw/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity&pitch/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^pitch/StatefulPartitionedCall^roll/StatefulPartitionedCall^yaw/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity%roll/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^pitch/StatefulPartitionedCall^roll/StatefulPartitionedCall^yaw/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????:::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
pitch/StatefulPartitionedCallpitch/StatefulPartitionedCall2<
roll/StatefulPartitionedCallroll/StatefulPartitionedCall2:
yaw/StatefulPartitionedCallyaw/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
C__inference_dense_layer_call_and_return_conditional_losses_71853319

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_71854283

inputs
identityh
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:?????????*
alpha%???=2
	LeakyReluo
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_71853379

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????x*
alpha%???=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????x:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
F__inference_conv1d_1_layer_call_and_return_conditional_losses_71854269

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_lambda_layer_call_and_return_conditional_losses_71853087

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slicen
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_4_layer_call_fn_71854455

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_718534182
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????Z:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_layer_call_fn_71854288

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_718532132
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_lambda_layer_call_fn_71854151

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_718530952
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
Y
-__inference_multiply_1_layer_call_fn_71854355
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_multiply_1_layer_call_and_return_conditional_losses_718532842
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_71853227

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_conv1d_2_layer_call_and_return_conditional_losses_71853151

inputs/
+conv1d_expanddims_1_readvariableop_resource
identity??"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
IdentityIdentityconv1d/Squeeze:output:0#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_71854421

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????x*
alpha%???=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????x:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?

*__inference_dense_2_layer_call_fn_71854445

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_718533972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
}
(__inference_dense_layer_call_fn_71854387

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_718533192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_71853786
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_718537452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
&__inference_signature_wrapper_71853831
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_718530092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
A__inference_yaw_layer_call_and_return_conditional_losses_71853488

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?

b
F__inference_lambda_2_layer_call_and_return_conditional_losses_71853054

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_sliceS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *?W?2
sub/yo
subSubstrided_slice:output:0sub/y:output:0*
T0*+
_output_shapes
:?????????2
subS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??G@2
mul/x`
mulMulmul/x:output:0sub:z:0*
T0*+
_output_shapes
:?????????2
mul_
IdentityIdentitymul:z:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

b
F__inference_lambda_2_layer_call_and_return_conditional_losses_71853066

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_sliceS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *?W?2
sub/yo
subSubstrided_slice:output:0sub/y:output:0*
T0*+
_output_shapes
:?????????2
subS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??G@2
mul/x`
mulMulmul/x:output:0sub:z:0*
T0*+
_output_shapes
:?????????2
mul_
IdentityIdentitymul:z:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_71854293

inputs
identityh
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:?????????*
alpha%???=2
	LeakyReluo
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_71853418

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????Z*
alpha%???=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????Z:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
`
D__inference_lambda_layer_call_and_return_conditional_losses_71854133

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slicen
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
Z
.__inference_concatenate_layer_call_fn_71854368
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_718533002
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
H
,__inference_flatten_1_layer_call_fn_71854331

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_718532272
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_2_layer_call_fn_71854180

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_718530542
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_71854326

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_71854407

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_71854392

inputs
identitye
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:??????????*
alpha%???=2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_conv1d_layer_call_fn_71854235

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_718531782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
C__inference_model_layer_call_and_return_conditional_losses_71853935

inputs8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource'
#roll_matmul_readvariableop_resource(
$roll_biasadd_readvariableop_resource(
$pitch_matmul_readvariableop_resource)
%pitch_biasadd_readvariableop_resource&
"yaw_matmul_readvariableop_resource'
#yaw_biasadd_readvariableop_resource
identity

identity_1

identity_2??conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?pitch/BiasAdd/ReadVariableOp?pitch/MatMul/ReadVariableOp?roll/BiasAdd/ReadVariableOp?roll/MatMul/ReadVariableOp?yaw/BiasAdd/ReadVariableOp?yaw/MatMul/ReadVariableOp?
lambda_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
lambda_1/strided_slice/stack?
lambda_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_1/strided_slice/stack_1?
lambda_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
lambda_1/strided_slice/stack_2?
lambda_1/strided_sliceStridedSliceinputs%lambda_1/strided_slice/stack:output:0'lambda_1/strided_slice/stack_1:output:0'lambda_1/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
lambda_1/strided_slice?
lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
lambda_2/strided_slice/stack?
lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_2/strided_slice/stack_1?
lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
lambda_2/strided_slice/stack_2?
lambda_2/strided_sliceStridedSliceinputs%lambda_2/strided_slice/stack:output:0'lambda_2/strided_slice/stack_1:output:0'lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
lambda_2/strided_slicee
lambda_2/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *?W?2
lambda_2/sub/y?
lambda_2/subSublambda_2/strided_slice:output:0lambda_2/sub/y:output:0*
T0*+
_output_shapes
:?????????2
lambda_2/sube
lambda_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??G@2
lambda_2/mul/x?
lambda_2/mulMullambda_2/mul/x:output:0lambda_2/sub:z:0*
T0*+
_output_shapes
:?????????2
lambda_2/mul?
lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lambda/strided_slice/stack?
lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
lambda/strided_slice/stack_1?
lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
lambda/strided_slice/stack_2?
lambda/strided_sliceStridedSliceinputs#lambda/strided_slice/stack:output:0%lambda/strided_slice/stack_1:output:0%lambda/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
lambda/strided_slice?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimslambda_1/strided_slice:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_1/BiasAdd?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimslambda_2/mul:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimslambda/strided_slice:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv1d_1/BiasAdd:output:0*+
_output_shapes
:?????????*
alpha%???=2
leaky_re_lu_1/LeakyRelu?
tf.math.sigmoid/SigmoidSigmoid conv1d_2/conv1d/Squeeze:output:0*
T0*+
_output_shapes
:?????????2
tf.math.sigmoid/Sigmoid?
leaky_re_lu/LeakyRelu	LeakyReluconv1d/BiasAdd:output:0*+
_output_shapes
:?????????*
alpha%???=2
leaky_re_lu/LeakyRelus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshape%leaky_re_lu_1/LeakyRelu:activations:0flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_1/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_2/Const?
flatten_2/ReshapeReshapetf.math.sigmoid/Sigmoid:y:0flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_2/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape#leaky_re_lu/LeakyRelu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten/Reshape?
multiply/mulMulflatten/Reshape:output:0flatten_2/Reshape:output:0*
T0*'
_output_shapes
:?????????2
multiply/mul?
multiply_1/mulMulflatten_1/Reshape:output:0flatten_2/Reshape:output:0*
T0*'
_output_shapes
:?????????2
multiply_1/mult
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2multiply/mul:z:0multiply_1/mul:z:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????22
concatenate/concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReludense/BiasAdd:output:0*(
_output_shapes
:??????????*
alpha%???=2
leaky_re_lu_2/LeakyRelu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul%leaky_re_lu_2/LeakyRelu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_1/BiasAdd?
leaky_re_lu_3/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*'
_output_shapes
:?????????x*
alpha%???=2
leaky_re_lu_3/LeakyRelu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:xZ*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul%leaky_re_lu_3/LeakyRelu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
dense_2/BiasAdd?
leaky_re_lu_4/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*'
_output_shapes
:?????????Z*
alpha%???=2
leaky_re_lu_4/LeakyRelu?
roll/MatMul/ReadVariableOpReadVariableOp#roll_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
roll/MatMul/ReadVariableOp?
roll/MatMulMatMul%leaky_re_lu_4/LeakyRelu:activations:0"roll/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
roll/MatMul?
roll/BiasAdd/ReadVariableOpReadVariableOp$roll_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
roll/BiasAdd/ReadVariableOp?
roll/BiasAddBiasAddroll/MatMul:product:0#roll/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
roll/BiasAdd?
pitch/MatMul/ReadVariableOpReadVariableOp$pitch_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
pitch/MatMul/ReadVariableOp?
pitch/MatMulMatMul%leaky_re_lu_4/LeakyRelu:activations:0#pitch/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
pitch/MatMul?
pitch/BiasAdd/ReadVariableOpReadVariableOp%pitch_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pitch/BiasAdd/ReadVariableOp?
pitch/BiasAddBiasAddpitch/MatMul:product:0$pitch/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
pitch/BiasAdd?
yaw/MatMul/ReadVariableOpReadVariableOp"yaw_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
yaw/MatMul/ReadVariableOp?

yaw/MatMulMatMul%leaky_re_lu_4/LeakyRelu:activations:0!yaw/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

yaw/MatMul?
yaw/BiasAdd/ReadVariableOpReadVariableOp#yaw_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
yaw/BiasAdd/ReadVariableOp?
yaw/BiasAddBiasAddyaw/MatMul:product:0"yaw/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
yaw/BiasAdd?
IdentityIdentityyaw/BiasAdd:output:0^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^pitch/BiasAdd/ReadVariableOp^pitch/MatMul/ReadVariableOp^roll/BiasAdd/ReadVariableOp^roll/MatMul/ReadVariableOp^yaw/BiasAdd/ReadVariableOp^yaw/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitypitch/BiasAdd:output:0^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^pitch/BiasAdd/ReadVariableOp^pitch/MatMul/ReadVariableOp^roll/BiasAdd/ReadVariableOp^roll/MatMul/ReadVariableOp^yaw/BiasAdd/ReadVariableOp^yaw/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityroll/BiasAdd:output:0^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^pitch/BiasAdd/ReadVariableOp^pitch/MatMul/ReadVariableOp^roll/BiasAdd/ReadVariableOp^roll/MatMul/ReadVariableOp^yaw/BiasAdd/ReadVariableOp^yaw/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????:::::::::::::::::2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2<
pitch/BiasAdd/ReadVariableOppitch/BiasAdd/ReadVariableOp2:
pitch/MatMul/ReadVariableOppitch/MatMul/ReadVariableOp2:
roll/BiasAdd/ReadVariableOproll/BiasAdd/ReadVariableOp28
roll/MatMul/ReadVariableOproll/MatMul/ReadVariableOp28
yaw/BiasAdd/ReadVariableOpyaw/BiasAdd/ReadVariableOp26
yaw/MatMul/ReadVariableOpyaw/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_2_layer_call_and_return_conditional_losses_71853241

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_1_layer_call_fn_71854211

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_718530292
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
W
+__inference_multiply_layer_call_fn_71854343
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_multiply_layer_call_and_return_conditional_losses_718532692
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?	
?
E__inference_dense_2_layer_call_and_return_conditional_losses_71854436

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xZ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_71853021

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slicen
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ҡ
?
#__inference__wrapped_model_71853009
input_1>
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource2
.model_conv1d_1_biasadd_readvariableop_resource>
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource<
8model_conv1d_conv1d_expanddims_1_readvariableop_resource0
,model_conv1d_biasadd_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource-
)model_roll_matmul_readvariableop_resource.
*model_roll_biasadd_readvariableop_resource.
*model_pitch_matmul_readvariableop_resource/
+model_pitch_biasadd_readvariableop_resource,
(model_yaw_matmul_readvariableop_resource-
)model_yaw_biasadd_readvariableop_resource
identity

identity_1

identity_2??#model/conv1d/BiasAdd/ReadVariableOp?/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp?%model/conv1d_1/BiasAdd/ReadVariableOp?1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?1model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?$model/dense_2/BiasAdd/ReadVariableOp?#model/dense_2/MatMul/ReadVariableOp?"model/pitch/BiasAdd/ReadVariableOp?!model/pitch/MatMul/ReadVariableOp?!model/roll/BiasAdd/ReadVariableOp? model/roll/MatMul/ReadVariableOp? model/yaw/BiasAdd/ReadVariableOp?model/yaw/MatMul/ReadVariableOp?
"model/lambda_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model/lambda_1/strided_slice/stack?
$model/lambda_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$model/lambda_1/strided_slice/stack_1?
$model/lambda_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$model/lambda_1/strided_slice/stack_2?
model/lambda_1/strided_sliceStridedSliceinput_1+model/lambda_1/strided_slice/stack:output:0-model/lambda_1/strided_slice/stack_1:output:0-model/lambda_1/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
model/lambda_1/strided_slice?
"model/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2$
"model/lambda_2/strided_slice/stack?
$model/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$model/lambda_2/strided_slice/stack_1?
$model/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$model/lambda_2/strided_slice/stack_2?
model/lambda_2/strided_sliceStridedSliceinput_1+model/lambda_2/strided_slice/stack:output:0-model/lambda_2/strided_slice/stack_1:output:0-model/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
model/lambda_2/strided_sliceq
model/lambda_2/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *?W?2
model/lambda_2/sub/y?
model/lambda_2/subSub%model/lambda_2/strided_slice:output:0model/lambda_2/sub/y:output:0*
T0*+
_output_shapes
:?????????2
model/lambda_2/subq
model/lambda_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??G@2
model/lambda_2/mul/x?
model/lambda_2/mulMulmodel/lambda_2/mul/x:output:0model/lambda_2/sub:z:0*
T0*+
_output_shapes
:?????????2
model/lambda_2/mul?
 model/lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 model/lambda/strided_slice/stack?
"model/lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2$
"model/lambda/strided_slice/stack_1?
"model/lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"model/lambda/strided_slice/stack_2?
model/lambda/strided_sliceStridedSliceinput_1)model/lambda/strided_slice/stack:output:0+model/lambda/strided_slice/stack_1:output:0+model/lambda/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
model/lambda/strided_slice?
$model/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/conv1d_1/conv1d/ExpandDims/dim?
 model/conv1d_1/conv1d/ExpandDims
ExpandDims%model/lambda_1/strided_slice:output:0-model/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2"
 model/conv1d_1/conv1d/ExpandDims?
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype023
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
&model/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/conv1d_1/conv1d/ExpandDims_1/dim?
"model/conv1d_1/conv1d/ExpandDims_1
ExpandDims9model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2$
"model/conv1d_1/conv1d/ExpandDims_1?
model/conv1d_1/conv1dConv2D)model/conv1d_1/conv1d/ExpandDims:output:0+model/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/conv1d_1/conv1d?
model/conv1d_1/conv1d/SqueezeSqueezemodel/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
model/conv1d_1/conv1d/Squeeze?
%model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv1d_1/BiasAdd/ReadVariableOp?
model/conv1d_1/BiasAddBiasAdd&model/conv1d_1/conv1d/Squeeze:output:0-model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
model/conv1d_1/BiasAdd?
$model/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/conv1d_2/conv1d/ExpandDims/dim?
 model/conv1d_2/conv1d/ExpandDims
ExpandDimsmodel/lambda_2/mul:z:0-model/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2"
 model/conv1d_2/conv1d/ExpandDims?
1model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype023
1model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
&model/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/conv1d_2/conv1d/ExpandDims_1/dim?
"model/conv1d_2/conv1d/ExpandDims_1
ExpandDims9model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2$
"model/conv1d_2/conv1d/ExpandDims_1?
model/conv1d_2/conv1dConv2D)model/conv1d_2/conv1d/ExpandDims:output:0+model/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/conv1d_2/conv1d?
model/conv1d_2/conv1d/SqueezeSqueezemodel/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
model/conv1d_2/conv1d/Squeeze?
"model/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model/conv1d/conv1d/ExpandDims/dim?
model/conv1d/conv1d/ExpandDims
ExpandDims#model/lambda/strided_slice:output:0+model/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2 
model/conv1d/conv1d/ExpandDims?
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype021
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
$model/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model/conv1d/conv1d/ExpandDims_1/dim?
 model/conv1d/conv1d/ExpandDims_1
ExpandDims7model/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2"
 model/conv1d/conv1d/ExpandDims_1?
model/conv1d/conv1dConv2D'model/conv1d/conv1d/ExpandDims:output:0)model/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/conv1d/conv1d?
model/conv1d/conv1d/SqueezeSqueezemodel/conv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
model/conv1d/conv1d/Squeeze?
#model/conv1d/BiasAdd/ReadVariableOpReadVariableOp,model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/conv1d/BiasAdd/ReadVariableOp?
model/conv1d/BiasAddBiasAdd$model/conv1d/conv1d/Squeeze:output:0+model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
model/conv1d/BiasAdd?
model/leaky_re_lu_1/LeakyRelu	LeakyRelumodel/conv1d_1/BiasAdd:output:0*+
_output_shapes
:?????????*
alpha%???=2
model/leaky_re_lu_1/LeakyRelu?
model/tf.math.sigmoid/SigmoidSigmoid&model/conv1d_2/conv1d/Squeeze:output:0*
T0*+
_output_shapes
:?????????2
model/tf.math.sigmoid/Sigmoid?
model/leaky_re_lu/LeakyRelu	LeakyRelumodel/conv1d/BiasAdd:output:0*+
_output_shapes
:?????????*
alpha%???=2
model/leaky_re_lu/LeakyRelu
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model/flatten_1/Const?
model/flatten_1/ReshapeReshape+model/leaky_re_lu_1/LeakyRelu:activations:0model/flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????2
model/flatten_1/Reshape
model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model/flatten_2/Const?
model/flatten_2/ReshapeReshape!model/tf.math.sigmoid/Sigmoid:y:0model/flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????2
model/flatten_2/Reshape{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model/flatten/Const?
model/flatten/ReshapeReshape)model/leaky_re_lu/LeakyRelu:activations:0model/flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
model/flatten/Reshape?
model/multiply/mulMulmodel/flatten/Reshape:output:0 model/flatten_2/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model/multiply/mul?
model/multiply_1/mulMul model/flatten_1/Reshape:output:0 model/flatten_2/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model/multiply_1/mul?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2model/multiply/mul:z:0model/multiply_1/mul:z:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????22
model/concatenate/concat?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense/BiasAdd?
model/leaky_re_lu_2/LeakyRelu	LeakyRelumodel/dense/BiasAdd:output:0*(
_output_shapes
:??????????*
alpha%???=2
model/leaky_re_lu_2/LeakyRelu?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02%
#model/dense_1/MatMul/ReadVariableOp?
model/dense_1/MatMulMatMul+model/leaky_re_lu_2/LeakyRelu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
model/dense_1/MatMul?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
model/dense_1/BiasAdd?
model/leaky_re_lu_3/LeakyRelu	LeakyRelumodel/dense_1/BiasAdd:output:0*'
_output_shapes
:?????????x*
alpha%???=2
model/leaky_re_lu_3/LeakyRelu?
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:xZ*
dtype02%
#model/dense_2/MatMul/ReadVariableOp?
model/dense_2/MatMulMatMul+model/leaky_re_lu_3/LeakyRelu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
model/dense_2/MatMul?
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
model/dense_2/BiasAdd?
model/leaky_re_lu_4/LeakyRelu	LeakyRelumodel/dense_2/BiasAdd:output:0*'
_output_shapes
:?????????Z*
alpha%???=2
model/leaky_re_lu_4/LeakyRelu?
 model/roll/MatMul/ReadVariableOpReadVariableOp)model_roll_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02"
 model/roll/MatMul/ReadVariableOp?
model/roll/MatMulMatMul+model/leaky_re_lu_4/LeakyRelu:activations:0(model/roll/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/roll/MatMul?
!model/roll/BiasAdd/ReadVariableOpReadVariableOp*model_roll_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/roll/BiasAdd/ReadVariableOp?
model/roll/BiasAddBiasAddmodel/roll/MatMul:product:0)model/roll/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/roll/BiasAdd?
!model/pitch/MatMul/ReadVariableOpReadVariableOp*model_pitch_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02#
!model/pitch/MatMul/ReadVariableOp?
model/pitch/MatMulMatMul+model/leaky_re_lu_4/LeakyRelu:activations:0)model/pitch/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/pitch/MatMul?
"model/pitch/BiasAdd/ReadVariableOpReadVariableOp+model_pitch_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/pitch/BiasAdd/ReadVariableOp?
model/pitch/BiasAddBiasAddmodel/pitch/MatMul:product:0*model/pitch/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/pitch/BiasAdd?
model/yaw/MatMul/ReadVariableOpReadVariableOp(model_yaw_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02!
model/yaw/MatMul/ReadVariableOp?
model/yaw/MatMulMatMul+model/leaky_re_lu_4/LeakyRelu:activations:0'model/yaw/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/yaw/MatMul?
 model/yaw/BiasAdd/ReadVariableOpReadVariableOp)model_yaw_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 model/yaw/BiasAdd/ReadVariableOp?
model/yaw/BiasAddBiasAddmodel/yaw/MatMul:product:0(model/yaw/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/yaw/BiasAdd?
IdentityIdentitymodel/pitch/BiasAdd:output:0$^model/conv1d/BiasAdd/ReadVariableOp0^model/conv1d/conv1d/ExpandDims_1/ReadVariableOp&^model/conv1d_1/BiasAdd/ReadVariableOp2^model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2^model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp#^model/pitch/BiasAdd/ReadVariableOp"^model/pitch/MatMul/ReadVariableOp"^model/roll/BiasAdd/ReadVariableOp!^model/roll/MatMul/ReadVariableOp!^model/yaw/BiasAdd/ReadVariableOp ^model/yaw/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitymodel/roll/BiasAdd:output:0$^model/conv1d/BiasAdd/ReadVariableOp0^model/conv1d/conv1d/ExpandDims_1/ReadVariableOp&^model/conv1d_1/BiasAdd/ReadVariableOp2^model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2^model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp#^model/pitch/BiasAdd/ReadVariableOp"^model/pitch/MatMul/ReadVariableOp"^model/roll/BiasAdd/ReadVariableOp!^model/roll/MatMul/ReadVariableOp!^model/yaw/BiasAdd/ReadVariableOp ^model/yaw/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitymodel/yaw/BiasAdd:output:0$^model/conv1d/BiasAdd/ReadVariableOp0^model/conv1d/conv1d/ExpandDims_1/ReadVariableOp&^model/conv1d_1/BiasAdd/ReadVariableOp2^model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2^model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp#^model/pitch/BiasAdd/ReadVariableOp"^model/pitch/MatMul/ReadVariableOp"^model/roll/BiasAdd/ReadVariableOp!^model/roll/MatMul/ReadVariableOp!^model/yaw/BiasAdd/ReadVariableOp ^model/yaw/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????:::::::::::::::::2J
#model/conv1d/BiasAdd/ReadVariableOp#model/conv1d/BiasAdd/ReadVariableOp2b
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp2N
%model/conv1d_1/BiasAdd/ReadVariableOp%model/conv1d_1/BiasAdd/ReadVariableOp2f
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2f
1model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp1model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2H
"model/pitch/BiasAdd/ReadVariableOp"model/pitch/BiasAdd/ReadVariableOp2F
!model/pitch/MatMul/ReadVariableOp!model/pitch/MatMul/ReadVariableOp2F
!model/roll/BiasAdd/ReadVariableOp!model/roll/BiasAdd/ReadVariableOp2D
 model/roll/MatMul/ReadVariableOp model/roll/MatMul/ReadVariableOp2D
 model/yaw/BiasAdd/ReadVariableOp model/yaw/BiasAdd/ReadVariableOp2B
model/yaw/MatMul/ReadVariableOpmodel/yaw/MatMul/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
??
?
C__inference_model_layer_call_and_return_conditional_losses_71854039

inputs8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource'
#roll_matmul_readvariableop_resource(
$roll_biasadd_readvariableop_resource(
$pitch_matmul_readvariableop_resource)
%pitch_biasadd_readvariableop_resource&
"yaw_matmul_readvariableop_resource'
#yaw_biasadd_readvariableop_resource
identity

identity_1

identity_2??conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?pitch/BiasAdd/ReadVariableOp?pitch/MatMul/ReadVariableOp?roll/BiasAdd/ReadVariableOp?roll/MatMul/ReadVariableOp?yaw/BiasAdd/ReadVariableOp?yaw/MatMul/ReadVariableOp?
lambda_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
lambda_1/strided_slice/stack?
lambda_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_1/strided_slice/stack_1?
lambda_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
lambda_1/strided_slice/stack_2?
lambda_1/strided_sliceStridedSliceinputs%lambda_1/strided_slice/stack:output:0'lambda_1/strided_slice/stack_1:output:0'lambda_1/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
lambda_1/strided_slice?
lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
lambda_2/strided_slice/stack?
lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_2/strided_slice/stack_1?
lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
lambda_2/strided_slice/stack_2?
lambda_2/strided_sliceStridedSliceinputs%lambda_2/strided_slice/stack:output:0'lambda_2/strided_slice/stack_1:output:0'lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
lambda_2/strided_slicee
lambda_2/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *?W?2
lambda_2/sub/y?
lambda_2/subSublambda_2/strided_slice:output:0lambda_2/sub/y:output:0*
T0*+
_output_shapes
:?????????2
lambda_2/sube
lambda_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??G@2
lambda_2/mul/x?
lambda_2/mulMullambda_2/mul/x:output:0lambda_2/sub:z:0*
T0*+
_output_shapes
:?????????2
lambda_2/mul?
lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lambda/strided_slice/stack?
lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
lambda/strided_slice/stack_1?
lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
lambda/strided_slice/stack_2?
lambda/strided_sliceStridedSliceinputs#lambda/strided_slice/stack:output:0%lambda/strided_slice/stack_1:output:0%lambda/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
lambda/strided_slice?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimslambda_1/strided_slice:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_1/BiasAdd?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimslambda_2/mul:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimslambda/strided_slice:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv1d_1/BiasAdd:output:0*+
_output_shapes
:?????????*
alpha%???=2
leaky_re_lu_1/LeakyRelu?
tf.math.sigmoid/SigmoidSigmoid conv1d_2/conv1d/Squeeze:output:0*
T0*+
_output_shapes
:?????????2
tf.math.sigmoid/Sigmoid?
leaky_re_lu/LeakyRelu	LeakyReluconv1d/BiasAdd:output:0*+
_output_shapes
:?????????*
alpha%???=2
leaky_re_lu/LeakyRelus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshape%leaky_re_lu_1/LeakyRelu:activations:0flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_1/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_2/Const?
flatten_2/ReshapeReshapetf.math.sigmoid/Sigmoid:y:0flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_2/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape#leaky_re_lu/LeakyRelu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten/Reshape?
multiply/mulMulflatten/Reshape:output:0flatten_2/Reshape:output:0*
T0*'
_output_shapes
:?????????2
multiply/mul?
multiply_1/mulMulflatten_1/Reshape:output:0flatten_2/Reshape:output:0*
T0*'
_output_shapes
:?????????2
multiply_1/mult
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2multiply/mul:z:0multiply_1/mul:z:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????22
concatenate/concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReludense/BiasAdd:output:0*(
_output_shapes
:??????????*
alpha%???=2
leaky_re_lu_2/LeakyRelu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul%leaky_re_lu_2/LeakyRelu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_1/BiasAdd?
leaky_re_lu_3/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*'
_output_shapes
:?????????x*
alpha%???=2
leaky_re_lu_3/LeakyRelu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:xZ*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul%leaky_re_lu_3/LeakyRelu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
dense_2/BiasAdd?
leaky_re_lu_4/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*'
_output_shapes
:?????????Z*
alpha%???=2
leaky_re_lu_4/LeakyRelu?
roll/MatMul/ReadVariableOpReadVariableOp#roll_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
roll/MatMul/ReadVariableOp?
roll/MatMulMatMul%leaky_re_lu_4/LeakyRelu:activations:0"roll/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
roll/MatMul?
roll/BiasAdd/ReadVariableOpReadVariableOp$roll_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
roll/BiasAdd/ReadVariableOp?
roll/BiasAddBiasAddroll/MatMul:product:0#roll/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
roll/BiasAdd?
pitch/MatMul/ReadVariableOpReadVariableOp$pitch_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
pitch/MatMul/ReadVariableOp?
pitch/MatMulMatMul%leaky_re_lu_4/LeakyRelu:activations:0#pitch/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
pitch/MatMul?
pitch/BiasAdd/ReadVariableOpReadVariableOp%pitch_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pitch/BiasAdd/ReadVariableOp?
pitch/BiasAddBiasAddpitch/MatMul:product:0$pitch/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
pitch/BiasAdd?
yaw/MatMul/ReadVariableOpReadVariableOp"yaw_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
yaw/MatMul/ReadVariableOp?

yaw/MatMulMatMul%leaky_re_lu_4/LeakyRelu:activations:0!yaw/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

yaw/MatMul?
yaw/BiasAdd/ReadVariableOpReadVariableOp#yaw_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
yaw/BiasAdd/ReadVariableOp?
yaw/BiasAddBiasAddyaw/MatMul:product:0"yaw/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
yaw/BiasAdd?
IdentityIdentityyaw/BiasAdd:output:0^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^pitch/BiasAdd/ReadVariableOp^pitch/MatMul/ReadVariableOp^roll/BiasAdd/ReadVariableOp^roll/MatMul/ReadVariableOp^yaw/BiasAdd/ReadVariableOp^yaw/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitypitch/BiasAdd:output:0^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^pitch/BiasAdd/ReadVariableOp^pitch/MatMul/ReadVariableOp^roll/BiasAdd/ReadVariableOp^roll/MatMul/ReadVariableOp^yaw/BiasAdd/ReadVariableOp^yaw/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityroll/BiasAdd:output:0^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^pitch/BiasAdd/ReadVariableOp^pitch/MatMul/ReadVariableOp^roll/BiasAdd/ReadVariableOp^roll/MatMul/ReadVariableOp^yaw/BiasAdd/ReadVariableOp^yaw/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????:::::::::::::::::2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2<
pitch/BiasAdd/ReadVariableOppitch/BiasAdd/ReadVariableOp2:
pitch/MatMul/ReadVariableOppitch/MatMul/ReadVariableOp2:
roll/BiasAdd/ReadVariableOproll/BiasAdd/ReadVariableOp28
roll/MatMul/ReadVariableOproll/MatMul/ReadVariableOp28
yaw/BiasAdd/ReadVariableOpyaw/BiasAdd/ReadVariableOp26
yaw/MatMul/ReadVariableOpyaw/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_lambda_layer_call_and_return_conditional_losses_71854141

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slicen
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_71853340

inputs
identitye
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:??????????*
alpha%???=2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?`
?
C__inference_model_layer_call_and_return_conditional_losses_71853571
input_1
conv1d_1_71853513
conv1d_1_71853515
conv1d_2_71853518
conv1d_71853521
conv1d_71853523
dense_71853535
dense_71853537
dense_1_71853541
dense_1_71853543
dense_2_71853547
dense_2_71853549
roll_71853553
roll_71853555
pitch_71853558
pitch_71853560
yaw_71853563
yaw_71853565
identity

identity_1

identity_2??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?pitch/StatefulPartitionedCall?roll/StatefulPartitionedCall?yaw/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_718530292
lambda_1/PartitionedCall?
lambda_2/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_718530662
lambda_2/PartitionedCall?
lambda/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_718530952
lambda/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_1_71853513conv1d_1_71853515*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_718531232"
 conv1d_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv1d_2_71853518*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_718531512"
 conv1d_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv1d_71853521conv1d_71853523*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_718531782 
conv1d/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_718531992
leaky_re_lu_1/PartitionedCall?
tf.math.sigmoid/SigmoidSigmoid)conv1d_2/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2
tf.math.sigmoid/Sigmoid?
leaky_re_lu/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_718532132
leaky_re_lu/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_718532272
flatten_1/PartitionedCall?
flatten_2/PartitionedCallPartitionedCalltf.math.sigmoid/Sigmoid:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_718532412
flatten_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_718532552
flatten/PartitionedCall?
multiply/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_multiply_layer_call_and_return_conditional_losses_718532692
multiply/PartitionedCall?
multiply_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_multiply_1_layer_call_and_return_conditional_losses_718532842
multiply_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0#multiply_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_718533002
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_71853535dense_71853537*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_718533192
dense/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_718533402
leaky_re_lu_2/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0dense_1_71853541dense_1_71853543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_718533582!
dense_1/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_718533792
leaky_re_lu_3/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0dense_2_71853547dense_2_71853549*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_718533972!
dense_2/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_718534182
leaky_re_lu_4/PartitionedCall?
roll/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0roll_71853553roll_71853555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_roll_layer_call_and_return_conditional_losses_718534362
roll/StatefulPartitionedCall?
pitch/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0pitch_71853558pitch_71853560*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_pitch_layer_call_and_return_conditional_losses_718534622
pitch/StatefulPartitionedCall?
yaw/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0yaw_71853563yaw_71853565*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_yaw_layer_call_and_return_conditional_losses_718534882
yaw/StatefulPartitionedCall?
IdentityIdentity$yaw/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^pitch/StatefulPartitionedCall^roll/StatefulPartitionedCall^yaw/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity&pitch/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^pitch/StatefulPartitionedCall^roll/StatefulPartitionedCall^yaw/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity%roll/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^pitch/StatefulPartitionedCall^roll/StatefulPartitionedCall^yaw/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????:::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
pitch/StatefulPartitionedCallpitch/StatefulPartitionedCall2<
roll/StatefulPartitionedCallroll/StatefulPartitionedCall2:
yaw/StatefulPartitionedCallyaw/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
L
0__inference_leaky_re_lu_3_layer_call_fn_71854426

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_718533792
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????x:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
q
+__inference_conv1d_2_layer_call_fn_71854254

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_718531512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_conv1d_1_layer_call_fn_71854278

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_718531232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_yaw_layer_call_and_return_conditional_losses_71854465

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_71853029

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slicen
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_71853358

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_conv1d_layer_call_and_return_conditional_losses_71853178

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_1_layer_call_fn_71854206

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_718530212
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?`
?
C__inference_model_layer_call_and_return_conditional_losses_71853745

inputs
conv1d_1_71853687
conv1d_1_71853689
conv1d_2_71853692
conv1d_71853695
conv1d_71853697
dense_71853709
dense_71853711
dense_1_71853715
dense_1_71853717
dense_2_71853721
dense_2_71853723
roll_71853727
roll_71853729
pitch_71853732
pitch_71853734
yaw_71853737
yaw_71853739
identity

identity_1

identity_2??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?pitch/StatefulPartitionedCall?roll/StatefulPartitionedCall?yaw/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_718530292
lambda_1/PartitionedCall?
lambda_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_718530662
lambda_2/PartitionedCall?
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_718530952
lambda/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_1_71853687conv1d_1_71853689*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_718531232"
 conv1d_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv1d_2_71853692*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_718531512"
 conv1d_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv1d_71853695conv1d_71853697*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_718531782 
conv1d/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_718531992
leaky_re_lu_1/PartitionedCall?
tf.math.sigmoid/SigmoidSigmoid)conv1d_2/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2
tf.math.sigmoid/Sigmoid?
leaky_re_lu/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_718532132
leaky_re_lu/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_718532272
flatten_1/PartitionedCall?
flatten_2/PartitionedCallPartitionedCalltf.math.sigmoid/Sigmoid:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_718532412
flatten_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_718532552
flatten/PartitionedCall?
multiply/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_multiply_layer_call_and_return_conditional_losses_718532692
multiply/PartitionedCall?
multiply_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_multiply_1_layer_call_and_return_conditional_losses_718532842
multiply_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0#multiply_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_718533002
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_71853709dense_71853711*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_718533192
dense/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_718533402
leaky_re_lu_2/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0dense_1_71853715dense_1_71853717*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_718533582!
dense_1/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_718533792
leaky_re_lu_3/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0dense_2_71853721dense_2_71853723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_718533972!
dense_2/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_718534182
leaky_re_lu_4/PartitionedCall?
roll/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0roll_71853727roll_71853729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_roll_layer_call_and_return_conditional_losses_718534362
roll/StatefulPartitionedCall?
pitch/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0pitch_71853732pitch_71853734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_pitch_layer_call_and_return_conditional_losses_718534622
pitch/StatefulPartitionedCall?
yaw/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0yaw_71853737yaw_71853739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_yaw_layer_call_and_return_conditional_losses_718534882
yaw/StatefulPartitionedCall?
IdentityIdentity$yaw/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^pitch/StatefulPartitionedCall^roll/StatefulPartitionedCall^yaw/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity&pitch/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^pitch/StatefulPartitionedCall^roll/StatefulPartitionedCall^yaw/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity%roll/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^pitch/StatefulPartitionedCall^roll/StatefulPartitionedCall^yaw/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????:::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
pitch/StatefulPartitionedCallpitch/StatefulPartitionedCall2<
roll/StatefulPartitionedCallroll/StatefulPartitionedCall2:
yaw/StatefulPartitionedCallyaw/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?`
?
C__inference_model_layer_call_and_return_conditional_losses_71853638

inputs
conv1d_1_71853580
conv1d_1_71853582
conv1d_2_71853585
conv1d_71853588
conv1d_71853590
dense_71853602
dense_71853604
dense_1_71853608
dense_1_71853610
dense_2_71853614
dense_2_71853616
roll_71853620
roll_71853622
pitch_71853625
pitch_71853627
yaw_71853630
yaw_71853632
identity

identity_1

identity_2??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?pitch/StatefulPartitionedCall?roll/StatefulPartitionedCall?yaw/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_718530212
lambda_1/PartitionedCall?
lambda_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_718530542
lambda_2/PartitionedCall?
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_718530872
lambda/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_1_71853580conv1d_1_71853582*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_718531232"
 conv1d_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv1d_2_71853585*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_718531512"
 conv1d_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv1d_71853588conv1d_71853590*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_718531782 
conv1d/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_718531992
leaky_re_lu_1/PartitionedCall?
tf.math.sigmoid/SigmoidSigmoid)conv1d_2/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2
tf.math.sigmoid/Sigmoid?
leaky_re_lu/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_718532132
leaky_re_lu/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_718532272
flatten_1/PartitionedCall?
flatten_2/PartitionedCallPartitionedCalltf.math.sigmoid/Sigmoid:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_718532412
flatten_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_718532552
flatten/PartitionedCall?
multiply/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_multiply_layer_call_and_return_conditional_losses_718532692
multiply/PartitionedCall?
multiply_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_multiply_1_layer_call_and_return_conditional_losses_718532842
multiply_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0#multiply_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_718533002
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_71853602dense_71853604*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_718533192
dense/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_718533402
leaky_re_lu_2/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0dense_1_71853608dense_1_71853610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_718533582!
dense_1/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_718533792
leaky_re_lu_3/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0dense_2_71853614dense_2_71853616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_718533972!
dense_2/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_718534182
leaky_re_lu_4/PartitionedCall?
roll/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0roll_71853620roll_71853622*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_roll_layer_call_and_return_conditional_losses_718534362
roll/StatefulPartitionedCall?
pitch/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0pitch_71853625pitch_71853627*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_pitch_layer_call_and_return_conditional_losses_718534622
pitch/StatefulPartitionedCall?
yaw/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0yaw_71853630yaw_71853632*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_yaw_layer_call_and_return_conditional_losses_718534882
yaw/StatefulPartitionedCall?
IdentityIdentity$yaw/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^pitch/StatefulPartitionedCall^roll/StatefulPartitionedCall^yaw/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity&pitch/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^pitch/StatefulPartitionedCall^roll/StatefulPartitionedCall^yaw/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity%roll/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^pitch/StatefulPartitionedCall^roll/StatefulPartitionedCall^yaw/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????:::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
pitch/StatefulPartitionedCallpitch/StatefulPartitionedCall2<
roll/StatefulPartitionedCallroll/StatefulPartitionedCall2:
yaw/StatefulPartitionedCallyaw/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_2_layer_call_and_return_conditional_losses_71854315

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_yaw_layer_call_fn_71854474

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_yaw_layer_call_and_return_conditional_losses_718534882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Z::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
H
,__inference_flatten_2_layer_call_fn_71854320

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_718532412
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_roll_layer_call_and_return_conditional_losses_71854503

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_2_layer_call_fn_71854397

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_718533402
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????9
pitch0
StatefulPartitionedCall:0?????????8
roll0
StatefulPartitionedCall:1?????????7
yaw0
StatefulPartitionedCall:2?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-3
layer-16
layer-17
layer_with_weights-4
layer-18
layer-19
layer_with_weights-5
layer-20
layer-21
layer_with_weights-6
layer-22
layer_with_weights-7
layer-23
layer_with_weights-8
layer-24
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMgAAAAdACgAXwAZABkAIUCZAFkAmQDhQNmAhkAdABq\nAqECUwCpBE7pAAAAAOkKAAAA6QIAAACpA9oCdGbaBGNhc3TaB2Zsb2F0MzKpAdoBa6kAcgsAAAD6\nPy9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25fZXh0ZW5zaW9uL21vZGVsL21v\nZGVscy5wedoIPGxhbWJkYT5oAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAEwAAAHMqAAAAdACgAWQBiAEbAHwAZABkAIUCZAJkA4UCZgIZ\nAIgAGAAUAHQAagKhAlMAqQRO6QEAAADpCgAAAOkPAAAAqQPaAnRm2gRjYXN02gdmbG9hdDMyqQHa\nAWupAtoEbWVhbtoDc3RkqQD6Py9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25f\nZXh0ZW5zaW9uL21vZGVsL21vZGVscy5wedoIPGxhbWJkYT5yAAAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [0.528683099669798, 0.3202638061825639]}]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_2", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMgAAAAdACgAXwAZABkAIUCZAFkAmQDhQNmAhkAdABq\nAqECUwCpBE7pAQAAAOkLAAAA6QIAAACpA9oCdGbaBGNhc3TaB2Zsb2F0MzKpAdoBa6kAcgsAAAD6\nPy9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25fZXh0ZW5zaW9uL21vZGVsL21v\nZGVscy5wedoIPGxhbWJkYT5tAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["lambda", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["lambda_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["lambda_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sigmoid", "trainable": true, "dtype": "float32", "function": "math.sigmoid"}, "name": "tf.math.sigmoid", "inbound_nodes": [["conv1d_2", 0, 0, {}]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_1", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["tf.math.sigmoid", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "multiply", "trainable": true, "dtype": "float32"}, "name": "multiply", "inbound_nodes": [[["flatten", 0, 0, {}], ["flatten_2", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "multiply_1", "trainable": true, "dtype": "float32"}, "name": "multiply_1", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["multiply", 0, 0, {}], ["multiply_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 150, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_2", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 120, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_3", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 90, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_4", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "yaw", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "yaw", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "pitch", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pitch", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "roll", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "roll", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["yaw", 0, 0], ["pitch", 0, 0], ["roll", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMgAAAAdACgAXwAZABkAIUCZAFkAmQDhQNmAhkAdABq\nAqECUwCpBE7pAAAAAOkKAAAA6QIAAACpA9oCdGbaBGNhc3TaB2Zsb2F0MzKpAdoBa6kAcgsAAAD6\nPy9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25fZXh0ZW5zaW9uL21vZGVsL21v\nZGVscy5wedoIPGxhbWJkYT5oAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAEwAAAHMqAAAAdACgAWQBiAEbAHwAZABkAIUCZAJkA4UCZgIZ\nAIgAGAAUAHQAagKhAlMAqQRO6QEAAADpCgAAAOkPAAAAqQPaAnRm2gRjYXN02gdmbG9hdDMyqQHa\nAWupAtoEbWVhbtoDc3RkqQD6Py9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25f\nZXh0ZW5zaW9uL21vZGVsL21vZGVscy5wedoIPGxhbWJkYT5yAAAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [0.528683099669798, 0.3202638061825639]}]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_2", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMgAAAAdACgAXwAZABkAIUCZAFkAmQDhQNmAhkAdABq\nAqECUwCpBE7pAQAAAOkLAAAA6QIAAACpA9oCdGbaBGNhc3TaB2Zsb2F0MzKpAdoBa6kAcgsAAAD6\nPy9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25fZXh0ZW5zaW9uL21vZGVsL21v\nZGVscy5wedoIPGxhbWJkYT5tAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["lambda", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["lambda_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["lambda_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sigmoid", "trainable": true, "dtype": "float32", "function": "math.sigmoid"}, "name": "tf.math.sigmoid", "inbound_nodes": [["conv1d_2", 0, 0, {}]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_1", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["tf.math.sigmoid", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "multiply", "trainable": true, "dtype": "float32"}, "name": "multiply", "inbound_nodes": [[["flatten", 0, 0, {}], ["flatten_2", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "multiply_1", "trainable": true, "dtype": "float32"}, "name": "multiply_1", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["multiply", 0, 0, {}], ["multiply_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 150, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_2", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 120, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_3", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 90, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_4", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "yaw", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "yaw", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "pitch", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pitch", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "roll", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "roll", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["yaw", 0, 0], ["pitch", 0, 0], ["roll", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
regularization_losses
 trainable_variables
!	variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMgAAAAdACgAXwAZABkAIUCZAFkAmQDhQNmAhkAdABq\nAqECUwCpBE7pAAAAAOkKAAAA6QIAAACpA9oCdGbaBGNhc3TaB2Zsb2F0MzKpAdoBa6kAcgsAAAD6\nPy9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25fZXh0ZW5zaW9uL21vZGVsL21v\nZGVscy5wedoIPGxhbWJkYT5oAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?
#regularization_losses
$trainable_variables
%	variables
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAEwAAAHMqAAAAdACgAWQBiAEbAHwAZABkAIUCZAJkA4UCZgIZ\nAIgAGAAUAHQAagKhAlMAqQRO6QEAAADpCgAAAOkPAAAAqQPaAnRm2gRjYXN02gdmbG9hdDMyqQHa\nAWupAtoEbWVhbtoDc3RkqQD6Py9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25f\nZXh0ZW5zaW9uL21vZGVsL21vZGVscy5wedoIPGxhbWJkYT5yAAAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [0.528683099669798, 0.3202638061825639]}]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?
'regularization_losses
(trainable_variables
)	variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMgAAAAdACgAXwAZABkAIUCZAFkAmQDhQNmAhkAdABq\nAqECUwCpBE7pAQAAAOkLAAAA6QIAAACpA9oCdGbaBGNhc3TaB2Zsb2F0MzKpAdoBa6kAcgsAAAD6\nPy9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25fZXh0ZW5zaW9uL21vZGVsL21v\nZGVscy5wedoIPGxhbWJkYT5tAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?	

+kernel
,bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 1]}}
?	

1kernel
2regularization_losses
3trainable_variables
4	variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 1]}}
?


6kernel
7bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 1]}}
?
<regularization_losses
=trainable_variables
>	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
?
@	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.sigmoid", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.sigmoid", "trainable": true, "dtype": "float32", "function": "math.sigmoid"}}
?
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
?
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Multiply", "name": "multiply", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multiply", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 25]}, {"class_name": "TensorShape", "items": [null, 25]}]}
?
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Multiply", "name": "multiply_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multiply_1", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 25]}, {"class_name": "TensorShape", "items": [null, 25]}]}
?
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 25]}, {"class_name": "TensorShape", "items": [null, 25]}]}
?

]kernel
^bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 150, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
?

gkernel
hbias
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 120, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150]}}
?
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
?

qkernel
rbias
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 90, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 120]}}
?
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
?

{kernel
|bias
}regularization_losses
~trainable_variables
	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "yaw", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "yaw", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 90}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90]}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "pitch", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pitch", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 90}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90]}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "roll", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "roll", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 90}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90]}}
 "
trackable_list_wrapper
?
+0
,1
12
63
74
]5
^6
g7
h8
q9
r10
{11
|12
?13
?14
?15
?16"
trackable_list_wrapper
?
+0
,1
12
63
74
]5
^6
g7
h8
q9
r10
{11
|12
?13
?14
?15
?16"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
regularization_losses
?layer_metrics
 ?layer_regularization_losses
trainable_variables
	variables
?layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
regularization_losses
?layer_metrics
 ?layer_regularization_losses
 trainable_variables
!	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
#regularization_losses
?layer_metrics
 ?layer_regularization_losses
$trainable_variables
%	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
'regularization_losses
?layer_metrics
 ?layer_regularization_losses
(trainable_variables
)	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!2conv1d/kernel
:2conv1d/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
-regularization_losses
?layer_metrics
 ?layer_regularization_losses
.trainable_variables
/	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2conv1d_2/kernel
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
'
10"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
2regularization_losses
?layer_metrics
 ?layer_regularization_losses
3trainable_variables
4	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2conv1d_1/kernel
:2conv1d_1/bias
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
8regularization_losses
?layer_metrics
 ?layer_regularization_losses
9trainable_variables
:	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
<regularization_losses
?layer_metrics
 ?layer_regularization_losses
=trainable_variables
>	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
Aregularization_losses
?layer_metrics
 ?layer_regularization_losses
Btrainable_variables
C	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
Eregularization_losses
?layer_metrics
 ?layer_regularization_losses
Ftrainable_variables
G	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
Iregularization_losses
?layer_metrics
 ?layer_regularization_losses
Jtrainable_variables
K	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
Mregularization_losses
?layer_metrics
 ?layer_regularization_losses
Ntrainable_variables
O	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
Qregularization_losses
?layer_metrics
 ?layer_regularization_losses
Rtrainable_variables
S	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
Uregularization_losses
?layer_metrics
 ?layer_regularization_losses
Vtrainable_variables
W	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
Yregularization_losses
?layer_metrics
 ?layer_regularization_losses
Ztrainable_variables
[	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	2?2dense/kernel
:?2
dense/bias
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
_regularization_losses
?layer_metrics
 ?layer_regularization_losses
`trainable_variables
a	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
cregularization_losses
?layer_metrics
 ?layer_regularization_losses
dtrainable_variables
e	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?x2dense_1/kernel
:x2dense_1/bias
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
iregularization_losses
?layer_metrics
 ?layer_regularization_losses
jtrainable_variables
k	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
mregularization_losses
?layer_metrics
 ?layer_regularization_losses
ntrainable_variables
o	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :xZ2dense_2/kernel
:Z2dense_2/bias
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
sregularization_losses
?layer_metrics
 ?layer_regularization_losses
ttrainable_variables
u	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
wregularization_losses
?layer_metrics
 ?layer_regularization_losses
xtrainable_variables
y	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:Z2
yaw/kernel
:2yaw/bias
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
}regularization_losses
?layer_metrics
 ?layer_regularization_losses
~trainable_variables
	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:Z2pitch/kernel
:2
pitch/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?trainable_variables
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:Z2roll/kernel
:2	roll/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?trainable_variables
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
C__inference_model_layer_call_and_return_conditional_losses_71854039
C__inference_model_layer_call_and_return_conditional_losses_71853935
C__inference_model_layer_call_and_return_conditional_losses_71853507
C__inference_model_layer_call_and_return_conditional_losses_71853571?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_71853009?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
input_1?????????
?2?
(__inference_model_layer_call_fn_71853679
(__inference_model_layer_call_fn_71854082
(__inference_model_layer_call_fn_71853786
(__inference_model_layer_call_fn_71854125?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_lambda_layer_call_and_return_conditional_losses_71854141
D__inference_lambda_layer_call_and_return_conditional_losses_71854133?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_lambda_layer_call_fn_71854151
)__inference_lambda_layer_call_fn_71854146?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_lambda_2_layer_call_and_return_conditional_losses_71854175
F__inference_lambda_2_layer_call_and_return_conditional_losses_71854163?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_lambda_2_layer_call_fn_71854180
+__inference_lambda_2_layer_call_fn_71854185?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_lambda_1_layer_call_and_return_conditional_losses_71854193
F__inference_lambda_1_layer_call_and_return_conditional_losses_71854201?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_lambda_1_layer_call_fn_71854211
+__inference_lambda_1_layer_call_fn_71854206?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv1d_layer_call_and_return_conditional_losses_71854226?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv1d_layer_call_fn_71854235?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv1d_2_layer_call_and_return_conditional_losses_71854247?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv1d_2_layer_call_fn_71854254?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv1d_1_layer_call_and_return_conditional_losses_71854269?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv1d_1_layer_call_fn_71854278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_71854283?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_layer_call_fn_71854288?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_71854293?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_leaky_re_lu_1_layer_call_fn_71854298?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_layer_call_and_return_conditional_losses_71854304?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_layer_call_fn_71854309?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_2_layer_call_and_return_conditional_losses_71854315?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_2_layer_call_fn_71854320?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_1_layer_call_and_return_conditional_losses_71854326?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_1_layer_call_fn_71854331?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_multiply_layer_call_and_return_conditional_losses_71854337?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_multiply_layer_call_fn_71854343?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_multiply_1_layer_call_and_return_conditional_losses_71854349?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_multiply_1_layer_call_fn_71854355?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_concatenate_layer_call_and_return_conditional_losses_71854362?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_concatenate_layer_call_fn_71854368?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_layer_call_and_return_conditional_losses_71854378?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_layer_call_fn_71854387?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_71854392?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_leaky_re_lu_2_layer_call_fn_71854397?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_1_layer_call_and_return_conditional_losses_71854407?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_1_layer_call_fn_71854416?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_71854421?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_leaky_re_lu_3_layer_call_fn_71854426?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_2_layer_call_and_return_conditional_losses_71854436?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_2_layer_call_fn_71854445?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_71854450?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_leaky_re_lu_4_layer_call_fn_71854455?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_yaw_layer_call_and_return_conditional_losses_71854465?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_yaw_layer_call_fn_71854474?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_pitch_layer_call_and_return_conditional_losses_71854484?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_pitch_layer_call_fn_71854493?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_roll_layer_call_and_return_conditional_losses_71854503?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_roll_layer_call_fn_71854512?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_71853831input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_71853009?671+,]^ghqr????{|4?1
*?'
%?"
input_1?????????
? "{?x
(
pitch?
pitch?????????
&
roll?
roll?????????
$
yaw?
yaw??????????
I__inference_concatenate_layer_call_and_return_conditional_losses_71854362?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????2
? ?
.__inference_concatenate_layer_call_fn_71854368vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "??????????2?
F__inference_conv1d_1_layer_call_and_return_conditional_losses_71854269d673?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
+__inference_conv1d_1_layer_call_fn_71854278W673?0
)?&
$?!
inputs?????????
? "???????????
F__inference_conv1d_2_layer_call_and_return_conditional_losses_71854247c13?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
+__inference_conv1d_2_layer_call_fn_71854254V13?0
)?&
$?!
inputs?????????
? "???????????
D__inference_conv1d_layer_call_and_return_conditional_losses_71854226d+,3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
)__inference_conv1d_layer_call_fn_71854235W+,3?0
)?&
$?!
inputs?????????
? "???????????
E__inference_dense_1_layer_call_and_return_conditional_losses_71854407]gh0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????x
? ~
*__inference_dense_1_layer_call_fn_71854416Pgh0?-
&?#
!?
inputs??????????
? "??????????x?
E__inference_dense_2_layer_call_and_return_conditional_losses_71854436\qr/?,
%?"
 ?
inputs?????????x
? "%?"
?
0?????????Z
? }
*__inference_dense_2_layer_call_fn_71854445Oqr/?,
%?"
 ?
inputs?????????x
? "??????????Z?
C__inference_dense_layer_call_and_return_conditional_losses_71854378]]^/?,
%?"
 ?
inputs?????????2
? "&?#
?
0??????????
? |
(__inference_dense_layer_call_fn_71854387P]^/?,
%?"
 ?
inputs?????????2
? "????????????
G__inference_flatten_1_layer_call_and_return_conditional_losses_71854326\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? 
,__inference_flatten_1_layer_call_fn_71854331O3?0
)?&
$?!
inputs?????????
? "???????????
G__inference_flatten_2_layer_call_and_return_conditional_losses_71854315\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? 
,__inference_flatten_2_layer_call_fn_71854320O3?0
)?&
$?!
inputs?????????
? "???????????
E__inference_flatten_layer_call_and_return_conditional_losses_71854304\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? }
*__inference_flatten_layer_call_fn_71854309O3?0
)?&
$?!
inputs?????????
? "???????????
F__inference_lambda_1_layer_call_and_return_conditional_losses_71854193h;?8
1?.
$?!
inputs?????????

 
p
? ")?&
?
0?????????
? ?
F__inference_lambda_1_layer_call_and_return_conditional_losses_71854201h;?8
1?.
$?!
inputs?????????

 
p 
? ")?&
?
0?????????
? ?
+__inference_lambda_1_layer_call_fn_71854206[;?8
1?.
$?!
inputs?????????

 
p
? "???????????
+__inference_lambda_1_layer_call_fn_71854211[;?8
1?.
$?!
inputs?????????

 
p 
? "???????????
F__inference_lambda_2_layer_call_and_return_conditional_losses_71854163h;?8
1?.
$?!
inputs?????????

 
p
? ")?&
?
0?????????
? ?
F__inference_lambda_2_layer_call_and_return_conditional_losses_71854175h;?8
1?.
$?!
inputs?????????

 
p 
? ")?&
?
0?????????
? ?
+__inference_lambda_2_layer_call_fn_71854180[;?8
1?.
$?!
inputs?????????

 
p
? "???????????
+__inference_lambda_2_layer_call_fn_71854185[;?8
1?.
$?!
inputs?????????

 
p 
? "???????????
D__inference_lambda_layer_call_and_return_conditional_losses_71854133h;?8
1?.
$?!
inputs?????????

 
p
? ")?&
?
0?????????
? ?
D__inference_lambda_layer_call_and_return_conditional_losses_71854141h;?8
1?.
$?!
inputs?????????

 
p 
? ")?&
?
0?????????
? ?
)__inference_lambda_layer_call_fn_71854146[;?8
1?.
$?!
inputs?????????

 
p
? "???????????
)__inference_lambda_layer_call_fn_71854151[;?8
1?.
$?!
inputs?????????

 
p 
? "???????????
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_71854293`3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
0__inference_leaky_re_lu_1_layer_call_fn_71854298S3?0
)?&
$?!
inputs?????????
? "???????????
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_71854392Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
0__inference_leaky_re_lu_2_layer_call_fn_71854397M0?-
&?#
!?
inputs??????????
? "????????????
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_71854421X/?,
%?"
 ?
inputs?????????x
? "%?"
?
0?????????x
? 
0__inference_leaky_re_lu_3_layer_call_fn_71854426K/?,
%?"
 ?
inputs?????????x
? "??????????x?
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_71854450X/?,
%?"
 ?
inputs?????????Z
? "%?"
?
0?????????Z
? 
0__inference_leaky_re_lu_4_layer_call_fn_71854455K/?,
%?"
 ?
inputs?????????Z
? "??????????Z?
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_71854283`3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
.__inference_leaky_re_lu_layer_call_fn_71854288S3?0
)?&
$?!
inputs?????????
? "???????????
C__inference_model_layer_call_and_return_conditional_losses_71853507?671+,]^ghqr????{|<?9
2?/
%?"
input_1?????????
p

 
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
C__inference_model_layer_call_and_return_conditional_losses_71853571?671+,]^ghqr????{|<?9
2?/
%?"
input_1?????????
p 

 
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
C__inference_model_layer_call_and_return_conditional_losses_71853935?671+,]^ghqr????{|;?8
1?.
$?!
inputs?????????
p

 
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
C__inference_model_layer_call_and_return_conditional_losses_71854039?671+,]^ghqr????{|;?8
1?.
$?!
inputs?????????
p 

 
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
(__inference_model_layer_call_fn_71853679?671+,]^ghqr????{|<?9
2?/
%?"
input_1?????????
p

 
? "Z?W
?
0?????????
?
1?????????
?
2??????????
(__inference_model_layer_call_fn_71853786?671+,]^ghqr????{|<?9
2?/
%?"
input_1?????????
p 

 
? "Z?W
?
0?????????
?
1?????????
?
2??????????
(__inference_model_layer_call_fn_71854082?671+,]^ghqr????{|;?8
1?.
$?!
inputs?????????
p

 
? "Z?W
?
0?????????
?
1?????????
?
2??????????
(__inference_model_layer_call_fn_71854125?671+,]^ghqr????{|;?8
1?.
$?!
inputs?????????
p 

 
? "Z?W
?
0?????????
?
1?????????
?
2??????????
H__inference_multiply_1_layer_call_and_return_conditional_losses_71854349?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
-__inference_multiply_1_layer_call_fn_71854355vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
F__inference_multiply_layer_call_and_return_conditional_losses_71854337?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
+__inference_multiply_layer_call_fn_71854343vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
C__inference_pitch_layer_call_and_return_conditional_losses_71854484^??/?,
%?"
 ?
inputs?????????Z
? "%?"
?
0?????????
? }
(__inference_pitch_layer_call_fn_71854493Q??/?,
%?"
 ?
inputs?????????Z
? "???????????
B__inference_roll_layer_call_and_return_conditional_losses_71854503^??/?,
%?"
 ?
inputs?????????Z
? "%?"
?
0?????????
? |
'__inference_roll_layer_call_fn_71854512Q??/?,
%?"
 ?
inputs?????????Z
? "???????????
&__inference_signature_wrapper_71853831?671+,]^ghqr????{|??<
? 
5?2
0
input_1%?"
input_1?????????"{?x
(
pitch?
pitch?????????
&
roll?
roll?????????
$
yaw?
yaw??????????
A__inference_yaw_layer_call_and_return_conditional_losses_71854465\{|/?,
%?"
 ?
inputs?????????Z
? "%?"
?
0?????????
? y
&__inference_yaw_layer_call_fn_71854474O{|/?,
%?"
 ?
inputs?????????Z
? "??????????