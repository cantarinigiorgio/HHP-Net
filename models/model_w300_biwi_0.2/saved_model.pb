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
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
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
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:22*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:2*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:2(*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:(*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:(*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
p

yaw/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
yaw/kernel
i
yaw/kernel/Read/ReadVariableOpReadVariableOp
yaw/kernel*
_output_shapes

:*
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
:*
shared_namepitch/kernel
m
 pitch/kernel/Read/ReadVariableOpReadVariableOppitch/kernel*
_output_shapes

:*
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
:*
shared_nameroll/kernel
k
roll/kernel/Read/ReadVariableOpReadVariableOproll/kernel*
_output_shapes

:*
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
R
	variables
 trainable_variables
!regularization_losses
"	keras_api
R
#	variables
$trainable_variables
%regularization_losses
&	keras_api
R
'	variables
(trainable_variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
^

1kernel
2	variables
3trainable_variables
4regularization_losses
5	keras_api
h

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
R
<	variables
=trainable_variables
>regularization_losses
?	keras_api

@	keras_api
R
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
R
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
R
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
R
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
R
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
R
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
R
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
h

]kernel
^bias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
R
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
h

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
R
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
h

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
R
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
i

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
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
 
?
?layers
	variables
?metrics
 ?layer_regularization_losses
trainable_variables
regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
 
?
?layers
	variables
?metrics
 ?layer_regularization_losses
 trainable_variables
!regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?layers
#	variables
?metrics
 ?layer_regularization_losses
$trainable_variables
%regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?layers
'	variables
?metrics
 ?layer_regularization_losses
(trainable_variables
)regularization_losses
?layer_metrics
?non_trainable_variables
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
?
?layers
-	variables
?metrics
 ?layer_regularization_losses
.trainable_variables
/regularization_losses
?layer_metrics
?non_trainable_variables
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE

10

10
 
?
?layers
2	variables
?metrics
 ?layer_regularization_losses
3trainable_variables
4regularization_losses
?layer_metrics
?non_trainable_variables
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
?
?layers
8	variables
?metrics
 ?layer_regularization_losses
9trainable_variables
:regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?layers
<	variables
?metrics
 ?layer_regularization_losses
=trainable_variables
>regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
 
?
?layers
A	variables
?metrics
 ?layer_regularization_losses
Btrainable_variables
Cregularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?layers
E	variables
?metrics
 ?layer_regularization_losses
Ftrainable_variables
Gregularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?layers
I	variables
?metrics
 ?layer_regularization_losses
Jtrainable_variables
Kregularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?layers
M	variables
?metrics
 ?layer_regularization_losses
Ntrainable_variables
Oregularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?layers
Q	variables
?metrics
 ?layer_regularization_losses
Rtrainable_variables
Sregularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?layers
U	variables
?metrics
 ?layer_regularization_losses
Vtrainable_variables
Wregularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?layers
Y	variables
?metrics
 ?layer_regularization_losses
Ztrainable_variables
[regularization_losses
?layer_metrics
?non_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

]0
^1

]0
^1
 
?
?layers
_	variables
?metrics
 ?layer_regularization_losses
`trainable_variables
aregularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?layers
c	variables
?metrics
 ?layer_regularization_losses
dtrainable_variables
eregularization_losses
?layer_metrics
?non_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
?
?layers
i	variables
?metrics
 ?layer_regularization_losses
jtrainable_variables
kregularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?layers
m	variables
?metrics
 ?layer_regularization_losses
ntrainable_variables
oregularization_losses
?layer_metrics
?non_trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1

q0
r1
 
?
?layers
s	variables
?metrics
 ?layer_regularization_losses
ttrainable_variables
uregularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?layers
w	variables
?metrics
 ?layer_regularization_losses
xtrainable_variables
yregularization_losses
?layer_metrics
?non_trainable_variables
VT
VARIABLE_VALUE
yaw/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEyaw/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1

{0
|1
 
?
?layers
}	variables
?metrics
 ?layer_regularization_losses
~trainable_variables
regularization_losses
?layer_metrics
?non_trainable_variables
XV
VARIABLE_VALUEpitch/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
pitch/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
?	variables
?metrics
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?layer_metrics
?non_trainable_variables
WU
VARIABLE_VALUEroll/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	roll/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
?	variables
?metrics
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?layer_metrics
?non_trainable_variables
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
&__inference_signature_wrapper_75890403
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
!__inference__traced_save_75891160
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
$__inference__traced_restore_75891221??
?
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_75889785

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
?
?
D__inference_conv1d_layer_call_and_return_conditional_losses_75890798

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
?
p
F__inference_multiply_layer_call_and_return_conditional_losses_75889841

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
̡
?
#__inference__wrapped_model_75889581
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

:22*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model/dense/BiasAdd?
model/leaky_re_lu_2/LeakyRelu	LeakyRelumodel/dense/BiasAdd:output:0*'
_output_shapes
:?????????2*
alpha%???=2
model/leaky_re_lu_2/LeakyRelu?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02%
#model/dense_1/MatMul/ReadVariableOp?
model/dense_1/MatMulMatMul+model/leaky_re_lu_2/LeakyRelu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model/dense_1/MatMul?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model/dense_1/BiasAdd?
model/leaky_re_lu_3/LeakyRelu	LeakyRelumodel/dense_1/BiasAdd:output:0*'
_output_shapes
:?????????(*
alpha%???=2
model/leaky_re_lu_3/LeakyRelu?
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02%
#model/dense_2/MatMul/ReadVariableOp?
model/dense_2/MatMulMatMul+model/leaky_re_lu_3/LeakyRelu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_2/MatMul?
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_2/BiasAdd?
model/leaky_re_lu_4/LeakyRelu	LeakyRelumodel/dense_2/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???=2
model/leaky_re_lu_4/LeakyRelu?
 model/roll/MatMul/ReadVariableOpReadVariableOp)model_roll_matmul_readvariableop_resource*
_output_shapes

:*
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

:*
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

:*
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
?
?
(__inference_model_layer_call_fn_75890654

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
C__inference_model_layer_call_and_return_conditional_losses_758902102
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
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_75889930

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

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
?
?
F__inference_conv1d_1_layer_call_and_return_conditional_losses_75889695

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
?
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_75889799

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
?
g
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_75889951

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????(*
alpha%???=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
s
I__inference_concatenate_layer_call_and_return_conditional_losses_75889872

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
?	
?
C__inference_pitch_layer_call_and_return_conditional_losses_75890034

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_2_layer_call_and_return_conditional_losses_75889813

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
D__inference_conv1d_layer_call_and_return_conditional_losses_75889750

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
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_75890765

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
?
J
.__inference_leaky_re_lu_layer_call_fn_75890860

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
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_758897852
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
)__inference_lambda_layer_call_fn_75890718

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
D__inference_lambda_layer_call_and_return_conditional_losses_758896592
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

b
F__inference_lambda_2_layer_call_and_return_conditional_losses_75890735

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
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_75889771

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
A__inference_yaw_layer_call_and_return_conditional_losses_75890060

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_1_layer_call_fn_75890783

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
F__inference_lambda_1_layer_call_and_return_conditional_losses_758896012
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
?
G
+__inference_lambda_2_layer_call_fn_75890752

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
F__inference_lambda_2_layer_call_and_return_conditional_losses_758896262
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
Z
.__inference_concatenate_layer_call_fn_75890940
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
I__inference_concatenate_layer_call_and_return_conditional_losses_758898722
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
?
?
+__inference_conv1d_1_layer_call_fn_75890850

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
F__inference_conv1d_1_layer_call_and_return_conditional_losses_758896952
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
?
r
F__inference_multiply_layer_call_and_return_conditional_losses_75890909
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
?
}
(__inference_dense_layer_call_fn_75890959

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
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_758898912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
q
+__inference_conv1d_2_layer_call_fn_75890826

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
F__inference_conv1d_2_layer_call_and_return_conditional_losses_758897232
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
?

b
F__inference_lambda_2_layer_call_and_return_conditional_losses_75889638

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
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_75889912

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????2*
alpha%???=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
}
(__inference_pitch_layer_call_fn_75891065

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
C__inference_pitch_layer_call_and_return_conditional_losses_758900342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_2_layer_call_and_return_conditional_losses_75891008

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_75890358
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
C__inference_model_layer_call_and_return_conditional_losses_758903172
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
?
g
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_75890865

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
?`
?
C__inference_model_layer_call_and_return_conditional_losses_75890143
input_1
conv1d_1_75890085
conv1d_1_75890087
conv1d_2_75890090
conv1d_75890093
conv1d_75890095
dense_75890107
dense_75890109
dense_1_75890113
dense_1_75890115
dense_2_75890119
dense_2_75890121
roll_75890125
roll_75890127
pitch_75890130
pitch_75890132
yaw_75890135
yaw_75890137
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
F__inference_lambda_1_layer_call_and_return_conditional_losses_758896012
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_758896382
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
D__inference_lambda_layer_call_and_return_conditional_losses_758896672
lambda/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_1_75890085conv1d_1_75890087*
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
F__inference_conv1d_1_layer_call_and_return_conditional_losses_758896952"
 conv1d_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv1d_2_75890090*
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
F__inference_conv1d_2_layer_call_and_return_conditional_losses_758897232"
 conv1d_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv1d_75890093conv1d_75890095*
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
D__inference_conv1d_layer_call_and_return_conditional_losses_758897502 
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
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_758897712
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
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_758897852
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
G__inference_flatten_1_layer_call_and_return_conditional_losses_758897992
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
G__inference_flatten_2_layer_call_and_return_conditional_losses_758898132
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
E__inference_flatten_layer_call_and_return_conditional_losses_758898272
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
F__inference_multiply_layer_call_and_return_conditional_losses_758898412
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
H__inference_multiply_1_layer_call_and_return_conditional_losses_758898562
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
I__inference_concatenate_layer_call_and_return_conditional_losses_758898722
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_75890107dense_75890109*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_758898912
dense/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_758899122
leaky_re_lu_2/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0dense_1_75890113dense_1_75890115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_758899302!
dense_1/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_758899512
leaky_re_lu_3/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0dense_2_75890119dense_2_75890121*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_758899692!
dense_2/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_758899902
leaky_re_lu_4/PartitionedCall?
roll/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0roll_75890125roll_75890127*
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
B__inference_roll_layer_call_and_return_conditional_losses_758900082
roll/StatefulPartitionedCall?
pitch/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0pitch_75890130pitch_75890132*
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
C__inference_pitch_layer_call_and_return_conditional_losses_758900342
pitch/StatefulPartitionedCall?
yaw/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0yaw_75890135yaw_75890137*
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
A__inference_yaw_layer_call_and_return_conditional_losses_758900602
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

b
F__inference_lambda_2_layer_call_and_return_conditional_losses_75890747

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
?
L
0__inference_leaky_re_lu_4_layer_call_fn_75891027

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_758899902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_yaw_layer_call_fn_75891046

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
A__inference_yaw_layer_call_and_return_conditional_losses_758900602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_2_layer_call_fn_75890892

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
G__inference_flatten_2_layer_call_and_return_conditional_losses_758898132
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
H
,__inference_flatten_1_layer_call_fn_75890903

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
G__inference_flatten_1_layer_call_and_return_conditional_losses_758897992
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
L
0__inference_leaky_re_lu_3_layer_call_fn_75890998

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
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_758899512
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_75890773

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
??
?
C__inference_model_layer_call_and_return_conditional_losses_75890611

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

:22*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReludense/BiasAdd:output:0*'
_output_shapes
:?????????2*
alpha%???=2
leaky_re_lu_2/LeakyRelu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul%leaky_re_lu_2/LeakyRelu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_1/BiasAdd?
leaky_re_lu_3/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*'
_output_shapes
:?????????(*
alpha%???=2
leaky_re_lu_3/LeakyRelu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul%leaky_re_lu_3/LeakyRelu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
leaky_re_lu_4/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???=2
leaky_re_lu_4/LeakyRelu?
roll/MatMul/ReadVariableOpReadVariableOp#roll_matmul_readvariableop_resource*
_output_shapes

:*
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

:*
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

:*
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
?+
?
!__inference__traced_save_75891160
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
?: ::::::22:2:2(:(:(:::::::: 2(
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
::$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2(: 	

_output_shapes
:(:$
 

_output_shapes

:(: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_75890898

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
?
g
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_75889990

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????*
alpha%???=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

b
F__inference_lambda_2_layer_call_and_return_conditional_losses_75889626

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
D__inference_lambda_layer_call_and_return_conditional_losses_75889667

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
G
+__inference_lambda_2_layer_call_fn_75890757

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
F__inference_lambda_2_layer_call_and_return_conditional_losses_758896382
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
?
`
D__inference_lambda_layer_call_and_return_conditional_losses_75890705

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

*__inference_dense_1_layer_call_fn_75890988

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
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_758899302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

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
(__inference_model_layer_call_fn_75890697

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
C__inference_model_layer_call_and_return_conditional_losses_758903172
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
?
?
F__inference_conv1d_1_layer_call_and_return_conditional_losses_75890841

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
?	
?
C__inference_pitch_layer_call_and_return_conditional_losses_75891056

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_75890876

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
B__inference_roll_layer_call_and_return_conditional_losses_75891075

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_1_layer_call_fn_75890778

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
F__inference_lambda_1_layer_call_and_return_conditional_losses_758895932
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
A__inference_yaw_layer_call_and_return_conditional_losses_75891037

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?`
?
C__inference_model_layer_call_and_return_conditional_losses_75890317

inputs
conv1d_1_75890259
conv1d_1_75890261
conv1d_2_75890264
conv1d_75890267
conv1d_75890269
dense_75890281
dense_75890283
dense_1_75890287
dense_1_75890289
dense_2_75890293
dense_2_75890295
roll_75890299
roll_75890301
pitch_75890304
pitch_75890306
yaw_75890309
yaw_75890311
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
F__inference_lambda_1_layer_call_and_return_conditional_losses_758896012
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_758896382
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
D__inference_lambda_layer_call_and_return_conditional_losses_758896672
lambda/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_1_75890259conv1d_1_75890261*
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
F__inference_conv1d_1_layer_call_and_return_conditional_losses_758896952"
 conv1d_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv1d_2_75890264*
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
F__inference_conv1d_2_layer_call_and_return_conditional_losses_758897232"
 conv1d_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv1d_75890267conv1d_75890269*
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
D__inference_conv1d_layer_call_and_return_conditional_losses_758897502 
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
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_758897712
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
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_758897852
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
G__inference_flatten_1_layer_call_and_return_conditional_losses_758897992
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
G__inference_flatten_2_layer_call_and_return_conditional_losses_758898132
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
E__inference_flatten_layer_call_and_return_conditional_losses_758898272
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
F__inference_multiply_layer_call_and_return_conditional_losses_758898412
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
H__inference_multiply_1_layer_call_and_return_conditional_losses_758898562
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
I__inference_concatenate_layer_call_and_return_conditional_losses_758898722
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_75890281dense_75890283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_758898912
dense/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_758899122
leaky_re_lu_2/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0dense_1_75890287dense_1_75890289*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_758899302!
dense_1/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_758899512
leaky_re_lu_3/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0dense_2_75890293dense_2_75890295*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_758899692!
dense_2/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_758899902
leaky_re_lu_4/PartitionedCall?
roll/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0roll_75890299roll_75890301*
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
B__inference_roll_layer_call_and_return_conditional_losses_758900082
roll/StatefulPartitionedCall?
pitch/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0pitch_75890304pitch_75890306*
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
C__inference_pitch_layer_call_and_return_conditional_losses_758900342
pitch/StatefulPartitionedCall?
yaw/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0yaw_75890309yaw_75890311*
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
A__inference_yaw_layer_call_and_return_conditional_losses_758900602
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
?
F
*__inference_flatten_layer_call_fn_75890881

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
E__inference_flatten_layer_call_and_return_conditional_losses_758898272
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
?
W
+__inference_multiply_layer_call_fn_75890915
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
F__inference_multiply_layer_call_and_return_conditional_losses_758898412
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
?`
?
C__inference_model_layer_call_and_return_conditional_losses_75890210

inputs
conv1d_1_75890152
conv1d_1_75890154
conv1d_2_75890157
conv1d_75890160
conv1d_75890162
dense_75890174
dense_75890176
dense_1_75890180
dense_1_75890182
dense_2_75890186
dense_2_75890188
roll_75890192
roll_75890194
pitch_75890197
pitch_75890199
yaw_75890202
yaw_75890204
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
F__inference_lambda_1_layer_call_and_return_conditional_losses_758895932
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_758896262
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
D__inference_lambda_layer_call_and_return_conditional_losses_758896592
lambda/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_1_75890152conv1d_1_75890154*
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
F__inference_conv1d_1_layer_call_and_return_conditional_losses_758896952"
 conv1d_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv1d_2_75890157*
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
F__inference_conv1d_2_layer_call_and_return_conditional_losses_758897232"
 conv1d_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv1d_75890160conv1d_75890162*
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
D__inference_conv1d_layer_call_and_return_conditional_losses_758897502 
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
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_758897712
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
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_758897852
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
G__inference_flatten_1_layer_call_and_return_conditional_losses_758897992
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
G__inference_flatten_2_layer_call_and_return_conditional_losses_758898132
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
E__inference_flatten_layer_call_and_return_conditional_losses_758898272
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
F__inference_multiply_layer_call_and_return_conditional_losses_758898412
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
H__inference_multiply_1_layer_call_and_return_conditional_losses_758898562
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
I__inference_concatenate_layer_call_and_return_conditional_losses_758898722
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_75890174dense_75890176*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_758898912
dense/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_758899122
leaky_re_lu_2/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0dense_1_75890180dense_1_75890182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_758899302!
dense_1/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_758899512
leaky_re_lu_3/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0dense_2_75890186dense_2_75890188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_758899692!
dense_2/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_758899902
leaky_re_lu_4/PartitionedCall?
roll/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0roll_75890192roll_75890194*
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
B__inference_roll_layer_call_and_return_conditional_losses_758900082
roll/StatefulPartitionedCall?
pitch/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0pitch_75890197pitch_75890199*
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
C__inference_pitch_layer_call_and_return_conditional_losses_758900342
pitch/StatefulPartitionedCall?
yaw/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0yaw_75890202yaw_75890204*
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
A__inference_yaw_layer_call_and_return_conditional_losses_758900602
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
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_75889593

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
?
r
H__inference_multiply_1_layer_call_and_return_conditional_losses_75889856

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
?
|
'__inference_roll_layer_call_fn_75891084

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
B__inference_roll_layer_call_and_return_conditional_losses_758900082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_75890855

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
?
L
0__inference_leaky_re_lu_1_layer_call_fn_75890870

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
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_758897712
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
?`
?
C__inference_model_layer_call_and_return_conditional_losses_75890079
input_1
conv1d_1_75889706
conv1d_1_75889708
conv1d_2_75889732
conv1d_75889761
conv1d_75889763
dense_75889902
dense_75889904
dense_1_75889941
dense_1_75889943
dense_2_75889980
dense_2_75889982
roll_75890019
roll_75890021
pitch_75890045
pitch_75890047
yaw_75890071
yaw_75890073
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
F__inference_lambda_1_layer_call_and_return_conditional_losses_758895932
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_758896262
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
D__inference_lambda_layer_call_and_return_conditional_losses_758896592
lambda/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_1_75889706conv1d_1_75889708*
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
F__inference_conv1d_1_layer_call_and_return_conditional_losses_758896952"
 conv1d_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv1d_2_75889732*
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
F__inference_conv1d_2_layer_call_and_return_conditional_losses_758897232"
 conv1d_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv1d_75889761conv1d_75889763*
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
D__inference_conv1d_layer_call_and_return_conditional_losses_758897502 
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
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_758897712
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
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_758897852
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
G__inference_flatten_1_layer_call_and_return_conditional_losses_758897992
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
G__inference_flatten_2_layer_call_and_return_conditional_losses_758898132
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
E__inference_flatten_layer_call_and_return_conditional_losses_758898272
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
F__inference_multiply_layer_call_and_return_conditional_losses_758898412
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
H__inference_multiply_1_layer_call_and_return_conditional_losses_758898562
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
I__inference_concatenate_layer_call_and_return_conditional_losses_758898722
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_75889902dense_75889904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_758898912
dense/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_758899122
leaky_re_lu_2/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0dense_1_75889941dense_1_75889943*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_758899302!
dense_1/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_758899512
leaky_re_lu_3/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0dense_2_75889980dense_2_75889982*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_758899692!
dense_2/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_758899902
leaky_re_lu_4/PartitionedCall?
roll/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0roll_75890019roll_75890021*
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
B__inference_roll_layer_call_and_return_conditional_losses_758900082
roll/StatefulPartitionedCall?
pitch/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0pitch_75890045pitch_75890047*
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
C__inference_pitch_layer_call_and_return_conditional_losses_758900342
pitch/StatefulPartitionedCall?
yaw/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0yaw_75890071yaw_75890073*
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
A__inference_yaw_layer_call_and_return_conditional_losses_758900602
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
F__inference_conv1d_2_layer_call_and_return_conditional_losses_75890819

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
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_75890979

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

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
t
H__inference_multiply_1_layer_call_and_return_conditional_losses_75890921
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
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_75889827

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
E__inference_dense_2_layer_call_and_return_conditional_losses_75889969

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_75889601

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
?
L
0__inference_leaky_re_lu_2_layer_call_fn_75890969

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
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_758899122
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?H
?
$__inference__traced_restore_75891221
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
?
u
I__inference_concatenate_layer_call_and_return_conditional_losses_75890934
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
?
?
&__inference_signature_wrapper_75890403
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
#__inference__wrapped_model_758895812
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
?
g
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_75890964

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????2*
alpha%???=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?

*__inference_dense_2_layer_call_fn_75891017

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_758899692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
F__inference_conv1d_2_layer_call_and_return_conditional_losses_75889723

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
?	
?
B__inference_roll_layer_call_and_return_conditional_losses_75890008

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_layer_call_and_return_conditional_losses_75889891

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

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
??
?
C__inference_model_layer_call_and_return_conditional_losses_75890507

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

:22*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReludense/BiasAdd:output:0*'
_output_shapes
:?????????2*
alpha%???=2
leaky_re_lu_2/LeakyRelu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul%leaky_re_lu_2/LeakyRelu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_1/BiasAdd?
leaky_re_lu_3/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*'
_output_shapes
:?????????(*
alpha%???=2
leaky_re_lu_3/LeakyRelu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul%leaky_re_lu_3/LeakyRelu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
leaky_re_lu_4/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???=2
leaky_re_lu_4/LeakyRelu?
roll/MatMul/ReadVariableOpReadVariableOp#roll_matmul_readvariableop_resource*
_output_shapes

:*
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

:*
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

:*
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
D__inference_lambda_layer_call_and_return_conditional_losses_75889659

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
Y
-__inference_multiply_1_layer_call_fn_75890927
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
H__inference_multiply_1_layer_call_and_return_conditional_losses_758898562
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
?
~
)__inference_conv1d_layer_call_fn_75890807

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
D__inference_conv1d_layer_call_and_return_conditional_losses_758897502
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
?
`
D__inference_lambda_layer_call_and_return_conditional_losses_75890713

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
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_75891022

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????*
alpha%???=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_lambda_layer_call_fn_75890723

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
D__inference_lambda_layer_call_and_return_conditional_losses_758896672
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
C__inference_dense_layer_call_and_return_conditional_losses_75890950

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

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
c
G__inference_flatten_2_layer_call_and_return_conditional_losses_75890887

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
?
g
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_75890993

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????(*
alpha%???=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_75890251
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
C__inference_model_layer_call_and_return_conditional_losses_758902102
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
_user_specified_name	input_1"?L
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMgAAAAdACgAXwAZABkAIUCZAFkAmQDhQNmAhkAdABq\nAqECUwCpBE7pAAAAAOkKAAAA6QIAAACpA9oCdGbaBGNhc3TaB2Zsb2F0MzKpAdoBa6kAcgsAAAD6\nPy9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25fZXh0ZW5zaW9uL21vZGVsL21v\nZGVscy5wedoIPGxhbWJkYT5oAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAEwAAAHMqAAAAdACgAWQBiAEbAHwAZABkAIUCZAJkA4UCZgIZ\nAIgAGAAUAHQAagKhAlMAqQRO6QEAAADpCgAAAOkPAAAAqQPaAnRm2gRjYXN02gdmbG9hdDMyqQHa\nAWupAtoEbWVhbtoDc3RkqQD6Py9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25f\nZXh0ZW5zaW9uL21vZGVsL21vZGVscy5wedoIPGxhbWJkYT5yAAAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [0.528683099669798, 0.3202638061825638]}]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_2", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMgAAAAdACgAXwAZABkAIUCZAFkAmQDhQNmAhkAdABq\nAqECUwCpBE7pAQAAAOkLAAAA6QIAAACpA9oCdGbaBGNhc3TaB2Zsb2F0MzKpAdoBa6kAcgsAAAD6\nPy9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25fZXh0ZW5zaW9uL21vZGVsL21v\nZGVscy5wedoIPGxhbWJkYT5tAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["lambda", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["lambda_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["lambda_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sigmoid", "trainable": true, "dtype": "float32", "function": "math.sigmoid"}, "name": "tf.math.sigmoid", "inbound_nodes": [["conv1d_2", 0, 0, {}]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_1", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["tf.math.sigmoid", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "multiply", "trainable": true, "dtype": "float32"}, "name": "multiply", "inbound_nodes": [[["flatten", 0, 0, {}], ["flatten_2", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "multiply_1", "trainable": true, "dtype": "float32"}, "name": "multiply_1", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["multiply", 0, 0, {}], ["multiply_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_2", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_3", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 30, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_4", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "yaw", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "yaw", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "pitch", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pitch", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "roll", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "roll", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["yaw", 0, 0], ["pitch", 0, 0], ["roll", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMgAAAAdACgAXwAZABkAIUCZAFkAmQDhQNmAhkAdABq\nAqECUwCpBE7pAAAAAOkKAAAA6QIAAACpA9oCdGbaBGNhc3TaB2Zsb2F0MzKpAdoBa6kAcgsAAAD6\nPy9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25fZXh0ZW5zaW9uL21vZGVsL21v\nZGVscy5wedoIPGxhbWJkYT5oAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAEwAAAHMqAAAAdACgAWQBiAEbAHwAZABkAIUCZAJkA4UCZgIZ\nAIgAGAAUAHQAagKhAlMAqQRO6QEAAADpCgAAAOkPAAAAqQPaAnRm2gRjYXN02gdmbG9hdDMyqQHa\nAWupAtoEbWVhbtoDc3RkqQD6Py9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25f\nZXh0ZW5zaW9uL21vZGVsL21vZGVscy5wedoIPGxhbWJkYT5yAAAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [0.528683099669798, 0.3202638061825638]}]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_2", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMgAAAAdACgAXwAZABkAIUCZAFkAmQDhQNmAhkAdABq\nAqECUwCpBE7pAQAAAOkLAAAA6QIAAACpA9oCdGbaBGNhc3TaB2Zsb2F0MzKpAdoBa6kAcgsAAAD6\nPy9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25fZXh0ZW5zaW9uL21vZGVsL21v\nZGVscy5wedoIPGxhbWJkYT5tAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["lambda", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["lambda_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["lambda_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sigmoid", "trainable": true, "dtype": "float32", "function": "math.sigmoid"}, "name": "tf.math.sigmoid", "inbound_nodes": [["conv1d_2", 0, 0, {}]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_1", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["tf.math.sigmoid", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "multiply", "trainable": true, "dtype": "float32"}, "name": "multiply", "inbound_nodes": [[["flatten", 0, 0, {}], ["flatten_2", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "multiply_1", "trainable": true, "dtype": "float32"}, "name": "multiply_1", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["multiply", 0, 0, {}], ["multiply_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_2", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_3", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 30, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}, "name": "leaky_re_lu_4", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "yaw", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "yaw", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "pitch", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pitch", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "roll", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "roll", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["yaw", 0, 0], ["pitch", 0, 0], ["roll", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMgAAAAdACgAXwAZABkAIUCZAFkAmQDhQNmAhkAdABq\nAqECUwCpBE7pAAAAAOkKAAAA6QIAAACpA9oCdGbaBGNhc3TaB2Zsb2F0MzKpAdoBa6kAcgsAAAD6\nPy9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25fZXh0ZW5zaW9uL21vZGVsL21v\nZGVscy5wedoIPGxhbWJkYT5oAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAEwAAAHMqAAAAdACgAWQBiAEbAHwAZABkAIUCZAJkA4UCZgIZ\nAIgAGAAUAHQAagKhAlMAqQRO6QEAAADpCgAAAOkPAAAAqQPaAnRm2gRjYXN02gdmbG9hdDMyqQHa\nAWupAtoEbWVhbtoDc3RkqQD6Py9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25f\nZXh0ZW5zaW9uL21vZGVsL21vZGVscy5wedoIPGxhbWJkYT5yAAAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [0.528683099669798, 0.3202638061825638]}]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 1]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMgAAAAdACgAXwAZABkAIUCZAFkAmQDhQNmAhkAdABq\nAqECUwCpBE7pAQAAAOkLAAAA6QIAAACpA9oCdGbaBGNhc3TaB2Zsb2F0MzKpAdoBa6kAcgsAAAD6\nPy9ob21lL2ltYXZpcy9Qcm9qZWN0cy9nYXplX2VzdGltYXRpb25fZXh0ZW5zaW9uL21vZGVsL21v\nZGVscy5wedoIPGxhbWJkYT5tAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "model.models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?	

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 1]}}
?	

1kernel
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 1]}}
?


6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 5, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 1]}}
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
?
@	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.sigmoid", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.sigmoid", "trainable": true, "dtype": "float32", "function": "math.sigmoid"}}
?
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
?
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Multiply", "name": "multiply", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multiply", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 25]}, {"class_name": "TensorShape", "items": [null, 25]}]}
?
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Multiply", "name": "multiply_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multiply_1", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 25]}, {"class_name": "TensorShape", "items": [null, 25]}]}
?
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 25]}, {"class_name": "TensorShape", "items": [null, 25]}]}
?

]kernel
^bias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
?

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
?

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 30, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
?

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "yaw", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "yaw", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "pitch", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pitch", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "roll", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "roll", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
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
 "
trackable_list_wrapper
?
?layers
	variables
?metrics
 ?layer_regularization_losses
trainable_variables
regularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
	variables
?metrics
 ?layer_regularization_losses
 trainable_variables
!regularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
#	variables
?metrics
 ?layer_regularization_losses
$trainable_variables
%regularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
'	variables
?metrics
 ?layer_regularization_losses
(trainable_variables
)regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!2conv1d/kernel
:2conv1d/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
-	variables
?metrics
 ?layer_regularization_losses
.trainable_variables
/regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2conv1d_2/kernel
'
10"
trackable_list_wrapper
'
10"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
2	variables
?metrics
 ?layer_regularization_losses
3trainable_variables
4regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2conv1d_1/kernel
:2conv1d_1/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
8	variables
?metrics
 ?layer_regularization_losses
9trainable_variables
:regularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
<	variables
?metrics
 ?layer_regularization_losses
=trainable_variables
>regularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
A	variables
?metrics
 ?layer_regularization_losses
Btrainable_variables
Cregularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
E	variables
?metrics
 ?layer_regularization_losses
Ftrainable_variables
Gregularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
I	variables
?metrics
 ?layer_regularization_losses
Jtrainable_variables
Kregularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
M	variables
?metrics
 ?layer_regularization_losses
Ntrainable_variables
Oregularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
Q	variables
?metrics
 ?layer_regularization_losses
Rtrainable_variables
Sregularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
U	variables
?metrics
 ?layer_regularization_losses
Vtrainable_variables
Wregularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
Y	variables
?metrics
 ?layer_regularization_losses
Ztrainable_variables
[regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:222dense/kernel
:22
dense/bias
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
_	variables
?metrics
 ?layer_regularization_losses
`trainable_variables
aregularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
c	variables
?metrics
 ?layer_regularization_losses
dtrainable_variables
eregularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2(2dense_1/kernel
:(2dense_1/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
i	variables
?metrics
 ?layer_regularization_losses
jtrainable_variables
kregularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
m	variables
?metrics
 ?layer_regularization_losses
ntrainable_variables
oregularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :(2dense_2/kernel
:2dense_2/bias
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
s	variables
?metrics
 ?layer_regularization_losses
ttrainable_variables
uregularization_losses
?layer_metrics
?non_trainable_variables
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
?layers
w	variables
?metrics
 ?layer_regularization_losses
xtrainable_variables
yregularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2
yaw/kernel
:2yaw/bias
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
}	variables
?metrics
 ?layer_regularization_losses
~trainable_variables
regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2pitch/kernel
:2
pitch/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
?metrics
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2roll/kernel
:2	roll/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
?metrics
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
#__inference__wrapped_model_75889581?
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
(__inference_model_layer_call_fn_75890654
(__inference_model_layer_call_fn_75890697
(__inference_model_layer_call_fn_75890358
(__inference_model_layer_call_fn_75890251?
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
?2?
C__inference_model_layer_call_and_return_conditional_losses_75890507
C__inference_model_layer_call_and_return_conditional_losses_75890611
C__inference_model_layer_call_and_return_conditional_losses_75890079
C__inference_model_layer_call_and_return_conditional_losses_75890143?
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
)__inference_lambda_layer_call_fn_75890723
)__inference_lambda_layer_call_fn_75890718?
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
D__inference_lambda_layer_call_and_return_conditional_losses_75890713
D__inference_lambda_layer_call_and_return_conditional_losses_75890705?
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
+__inference_lambda_2_layer_call_fn_75890752
+__inference_lambda_2_layer_call_fn_75890757?
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_75890747
F__inference_lambda_2_layer_call_and_return_conditional_losses_75890735?
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
+__inference_lambda_1_layer_call_fn_75890778
+__inference_lambda_1_layer_call_fn_75890783?
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
F__inference_lambda_1_layer_call_and_return_conditional_losses_75890765
F__inference_lambda_1_layer_call_and_return_conditional_losses_75890773?
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
)__inference_conv1d_layer_call_fn_75890807?
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
D__inference_conv1d_layer_call_and_return_conditional_losses_75890798?
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
+__inference_conv1d_2_layer_call_fn_75890826?
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
F__inference_conv1d_2_layer_call_and_return_conditional_losses_75890819?
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
+__inference_conv1d_1_layer_call_fn_75890850?
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
F__inference_conv1d_1_layer_call_and_return_conditional_losses_75890841?
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
.__inference_leaky_re_lu_layer_call_fn_75890860?
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
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_75890855?
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
0__inference_leaky_re_lu_1_layer_call_fn_75890870?
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
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_75890865?
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
*__inference_flatten_layer_call_fn_75890881?
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
E__inference_flatten_layer_call_and_return_conditional_losses_75890876?
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
,__inference_flatten_2_layer_call_fn_75890892?
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
G__inference_flatten_2_layer_call_and_return_conditional_losses_75890887?
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
,__inference_flatten_1_layer_call_fn_75890903?
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
G__inference_flatten_1_layer_call_and_return_conditional_losses_75890898?
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
+__inference_multiply_layer_call_fn_75890915?
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
F__inference_multiply_layer_call_and_return_conditional_losses_75890909?
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
-__inference_multiply_1_layer_call_fn_75890927?
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
H__inference_multiply_1_layer_call_and_return_conditional_losses_75890921?
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
.__inference_concatenate_layer_call_fn_75890940?
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
I__inference_concatenate_layer_call_and_return_conditional_losses_75890934?
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
(__inference_dense_layer_call_fn_75890959?
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
C__inference_dense_layer_call_and_return_conditional_losses_75890950?
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
0__inference_leaky_re_lu_2_layer_call_fn_75890969?
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
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_75890964?
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
*__inference_dense_1_layer_call_fn_75890988?
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
E__inference_dense_1_layer_call_and_return_conditional_losses_75890979?
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
0__inference_leaky_re_lu_3_layer_call_fn_75890998?
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
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_75890993?
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
*__inference_dense_2_layer_call_fn_75891017?
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
E__inference_dense_2_layer_call_and_return_conditional_losses_75891008?
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
0__inference_leaky_re_lu_4_layer_call_fn_75891027?
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
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_75891022?
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
&__inference_yaw_layer_call_fn_75891046?
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
A__inference_yaw_layer_call_and_return_conditional_losses_75891037?
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
(__inference_pitch_layer_call_fn_75891065?
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
C__inference_pitch_layer_call_and_return_conditional_losses_75891056?
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
'__inference_roll_layer_call_fn_75891084?
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
B__inference_roll_layer_call_and_return_conditional_losses_75891075?
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
&__inference_signature_wrapper_75890403input_1"?
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
#__inference__wrapped_model_75889581?671+,]^ghqr????{|4?1
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
I__inference_concatenate_layer_call_and_return_conditional_losses_75890934?Z?W
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
.__inference_concatenate_layer_call_fn_75890940vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "??????????2?
F__inference_conv1d_1_layer_call_and_return_conditional_losses_75890841d673?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
+__inference_conv1d_1_layer_call_fn_75890850W673?0
)?&
$?!
inputs?????????
? "???????????
F__inference_conv1d_2_layer_call_and_return_conditional_losses_75890819c13?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
+__inference_conv1d_2_layer_call_fn_75890826V13?0
)?&
$?!
inputs?????????
? "???????????
D__inference_conv1d_layer_call_and_return_conditional_losses_75890798d+,3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
)__inference_conv1d_layer_call_fn_75890807W+,3?0
)?&
$?!
inputs?????????
? "???????????
E__inference_dense_1_layer_call_and_return_conditional_losses_75890979\gh/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????(
? }
*__inference_dense_1_layer_call_fn_75890988Ogh/?,
%?"
 ?
inputs?????????2
? "??????????(?
E__inference_dense_2_layer_call_and_return_conditional_losses_75891008\qr/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? }
*__inference_dense_2_layer_call_fn_75891017Oqr/?,
%?"
 ?
inputs?????????(
? "???????????
C__inference_dense_layer_call_and_return_conditional_losses_75890950\]^/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? {
(__inference_dense_layer_call_fn_75890959O]^/?,
%?"
 ?
inputs?????????2
? "??????????2?
G__inference_flatten_1_layer_call_and_return_conditional_losses_75890898\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? 
,__inference_flatten_1_layer_call_fn_75890903O3?0
)?&
$?!
inputs?????????
? "???????????
G__inference_flatten_2_layer_call_and_return_conditional_losses_75890887\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? 
,__inference_flatten_2_layer_call_fn_75890892O3?0
)?&
$?!
inputs?????????
? "???????????
E__inference_flatten_layer_call_and_return_conditional_losses_75890876\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? }
*__inference_flatten_layer_call_fn_75890881O3?0
)?&
$?!
inputs?????????
? "???????????
F__inference_lambda_1_layer_call_and_return_conditional_losses_75890765h;?8
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
F__inference_lambda_1_layer_call_and_return_conditional_losses_75890773h;?8
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
+__inference_lambda_1_layer_call_fn_75890778[;?8
1?.
$?!
inputs?????????

 
p
? "???????????
+__inference_lambda_1_layer_call_fn_75890783[;?8
1?.
$?!
inputs?????????

 
p 
? "???????????
F__inference_lambda_2_layer_call_and_return_conditional_losses_75890735h;?8
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_75890747h;?8
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
+__inference_lambda_2_layer_call_fn_75890752[;?8
1?.
$?!
inputs?????????

 
p
? "???????????
+__inference_lambda_2_layer_call_fn_75890757[;?8
1?.
$?!
inputs?????????

 
p 
? "???????????
D__inference_lambda_layer_call_and_return_conditional_losses_75890705h;?8
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
D__inference_lambda_layer_call_and_return_conditional_losses_75890713h;?8
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
)__inference_lambda_layer_call_fn_75890718[;?8
1?.
$?!
inputs?????????

 
p
? "???????????
)__inference_lambda_layer_call_fn_75890723[;?8
1?.
$?!
inputs?????????

 
p 
? "???????????
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_75890865`3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
0__inference_leaky_re_lu_1_layer_call_fn_75890870S3?0
)?&
$?!
inputs?????????
? "???????????
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_75890964X/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? 
0__inference_leaky_re_lu_2_layer_call_fn_75890969K/?,
%?"
 ?
inputs?????????2
? "??????????2?
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_75890993X/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????(
? 
0__inference_leaky_re_lu_3_layer_call_fn_75890998K/?,
%?"
 ?
inputs?????????(
? "??????????(?
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_75891022X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
0__inference_leaky_re_lu_4_layer_call_fn_75891027K/?,
%?"
 ?
inputs?????????
? "???????????
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_75890855`3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
.__inference_leaky_re_lu_layer_call_fn_75890860S3?0
)?&
$?!
inputs?????????
? "???????????
C__inference_model_layer_call_and_return_conditional_losses_75890079?671+,]^ghqr????{|<?9
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
C__inference_model_layer_call_and_return_conditional_losses_75890143?671+,]^ghqr????{|<?9
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
C__inference_model_layer_call_and_return_conditional_losses_75890507?671+,]^ghqr????{|;?8
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
C__inference_model_layer_call_and_return_conditional_losses_75890611?671+,]^ghqr????{|;?8
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
(__inference_model_layer_call_fn_75890251?671+,]^ghqr????{|<?9
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
(__inference_model_layer_call_fn_75890358?671+,]^ghqr????{|<?9
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
(__inference_model_layer_call_fn_75890654?671+,]^ghqr????{|;?8
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
(__inference_model_layer_call_fn_75890697?671+,]^ghqr????{|;?8
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
H__inference_multiply_1_layer_call_and_return_conditional_losses_75890921?Z?W
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
-__inference_multiply_1_layer_call_fn_75890927vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
F__inference_multiply_layer_call_and_return_conditional_losses_75890909?Z?W
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
+__inference_multiply_layer_call_fn_75890915vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
C__inference_pitch_layer_call_and_return_conditional_losses_75891056^??/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
(__inference_pitch_layer_call_fn_75891065Q??/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_roll_layer_call_and_return_conditional_losses_75891075^??/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
'__inference_roll_layer_call_fn_75891084Q??/?,
%?"
 ?
inputs?????????
? "???????????
&__inference_signature_wrapper_75890403?671+,]^ghqr????{|??<
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
A__inference_yaw_layer_call_and_return_conditional_losses_75891037\{|/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_yaw_layer_call_fn_75891046O{|/?,
%?"
 ?
inputs?????????
? "??????????