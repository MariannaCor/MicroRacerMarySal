��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
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
delete_old_dirsbool(�
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
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
@
Softplus
features"T
activations"T"
Ttype:
2
�
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.02v2.6.0-rc2-32-g919f693420e8��
y
layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namelayer_1/kernel
r
"layer_1/kernel/Read/ReadVariableOpReadVariableOplayer_1/kernel*
_output_shapes
:	�*
dtype0
q
layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelayer_1/bias
j
 layer_1/bias/Read/ReadVariableOpReadVariableOplayer_1/bias*
_output_shapes	
:�*
dtype0
z
layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namelayer_2/kernel
s
"layer_2/kernel/Read/ReadVariableOpReadVariableOplayer_2/kernel* 
_output_shapes
:
��*
dtype0
q
layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelayer_2/bias
j
 layer_2/bias/Read/ReadVariableOpReadVariableOplayer_2/bias*
_output_shapes	
:�*
dtype0
z
layer_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namelayer_3/kernel
s
"layer_3/kernel/Read/ReadVariableOpReadVariableOplayer_3/kernel* 
_output_shapes
:
��*
dtype0
q
layer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelayer_3/bias
j
 layer_3/bias/Read/ReadVariableOpReadVariableOplayer_3/bias*
_output_shapes	
:�*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/layer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/layer_1/kernel/m
�
)Adam/layer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_1/kernel/m*
_output_shapes
:	�*
dtype0

Adam/layer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/layer_1/bias/m
x
'Adam/layer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/layer_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/layer_2/kernel/m
�
)Adam/layer_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_2/kernel/m* 
_output_shapes
:
��*
dtype0

Adam/layer_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/layer_2/bias/m
x
'Adam/layer_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/layer_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/layer_3/kernel/m
�
)Adam/layer_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_3/kernel/m* 
_output_shapes
:
��*
dtype0

Adam/layer_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/layer_3/bias/m
x
'Adam/layer_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_3/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	�*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
�
Adam/layer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/layer_1/kernel/v
�
)Adam/layer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_1/kernel/v*
_output_shapes
:	�*
dtype0

Adam/layer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/layer_1/bias/v
x
'Adam/layer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/layer_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/layer_2/kernel/v
�
)Adam/layer_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_2/kernel/v* 
_output_shapes
:
��*
dtype0

Adam/layer_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/layer_2/bias/v
x
'Adam/layer_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/layer_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/layer_3/kernel/v
�
)Adam/layer_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_3/kernel/v* 
_output_shapes
:
��*
dtype0

Adam/layer_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/layer_3/bias/v
x
'Adam/layer_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_3/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	�*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�-
value�-B�- B�-
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer

signatures
#_self_saveable_object_factories
	regularization_losses

trainable_variables
	variables
	keras_api
<
_init_input_shape
#_self_saveable_object_factories
�

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
�

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
�

kernel
bias
#_self_saveable_object_factories
 regularization_losses
!trainable_variables
"	variables
#	keras_api
�

$kernel
%bias
#&_self_saveable_object_factories
'regularization_losses
(trainable_variables
)	variables
*	keras_api
�
+iter

,beta_1

-beta_2
	.decay
/learning_ratemNmOmPmQmRmS$mT%mUvVvWvXvYvZv[$v\%v]
 
 
 
8
0
1
2
3
4
5
$6
%7
8
0
1
2
3
4
5
$6
%7
�
	regularization_losses
0layer_metrics

1layers
2non_trainable_variables

trainable_variables
3metrics
4layer_regularization_losses
	variables
 
 
ZX
VARIABLE_VALUElayer_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
�
regularization_losses
5layer_metrics

6layers
7metrics
8non_trainable_variables
trainable_variables
9layer_regularization_losses
	variables
ZX
VARIABLE_VALUElayer_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
�
regularization_losses
:layer_metrics

;layers
<metrics
=non_trainable_variables
trainable_variables
>layer_regularization_losses
	variables
ZX
VARIABLE_VALUElayer_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
�
 regularization_losses
?layer_metrics

@layers
Ametrics
Bnon_trainable_variables
!trainable_variables
Clayer_regularization_losses
"	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

$0
%1

$0
%1
�
'regularization_losses
Dlayer_metrics

Elayers
Fmetrics
Gnon_trainable_variables
(trainable_variables
Hlayer_regularization_losses
)	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4
 

I0
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
4
	Jtotal
	Kcount
L	variables
M	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

L	variables
}{
VARIABLE_VALUEAdam/layer_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_2Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2layer_1/kernellayer_1/biaslayer_2/kernellayer_2/biaslayer_3/kernellayer_3/biasdense/kernel
dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*1
config_proto!

CPU

GPU 
(
2J 8� */
f*R(
&__inference_signature_wrapper_24935574
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"layer_1/kernel/Read/ReadVariableOp layer_1/bias/Read/ReadVariableOp"layer_2/kernel/Read/ReadVariableOp layer_2/bias/Read/ReadVariableOp"layer_3/kernel/Read/ReadVariableOp layer_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/layer_1/kernel/m/Read/ReadVariableOp'Adam/layer_1/bias/m/Read/ReadVariableOp)Adam/layer_2/kernel/m/Read/ReadVariableOp'Adam/layer_2/bias/m/Read/ReadVariableOp)Adam/layer_3/kernel/m/Read/ReadVariableOp'Adam/layer_3/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/layer_1/kernel/v/Read/ReadVariableOp'Adam/layer_1/bias/v/Read/ReadVariableOp)Adam/layer_2/kernel/v/Read/ReadVariableOp'Adam/layer_2/bias/v/Read/ReadVariableOp)Adam/layer_3/kernel/v/Read/ReadVariableOp'Adam/layer_3/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU 
(
2J 8� **
f%R#
!__inference__traced_save_24935876
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_1/kernellayer_1/biaslayer_2/kernellayer_2/biaslayer_3/kernellayer_3/biasdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/layer_1/kernel/mAdam/layer_1/bias/mAdam/layer_2/kernel/mAdam/layer_2/bias/mAdam/layer_3/kernel/mAdam/layer_3/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/layer_1/kernel/vAdam/layer_1/bias/vAdam/layer_2/kernel/vAdam/layer_2/bias/vAdam/layer_3/kernel/vAdam/layer_3/bias/vAdam/dense/kernel/vAdam/dense/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU 
(
2J 8� *-
f(R&
$__inference__traced_restore_24935979��
�
�
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935457

inputs#
layer_1_24935436:	�
layer_1_24935438:	�$
layer_2_24935441:
��
layer_2_24935443:	�$
layer_3_24935446:
��
layer_3_24935448:	�!
dense_24935451:	�
dense_24935453:
identity��dense/StatefulPartitionedCall�layer_1/StatefulPartitionedCall�layer_2/StatefulPartitionedCall�layer_3/StatefulPartitionedCall�
layer_1/StatefulPartitionedCallStatefulPartitionedCallinputslayer_1_24935436layer_1_24935438*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_1_layer_call_and_return_conditional_losses_249352932!
layer_1/StatefulPartitionedCall�
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_24935441layer_2_24935443*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_2_layer_call_and_return_conditional_losses_249353102!
layer_2/StatefulPartitionedCall�
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_24935446layer_3_24935448*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_3_layer_call_and_return_conditional_losses_249353272!
layer_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall(layer_3/StatefulPartitionedCall:output:0dense_24935451dense_24935453*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_249353442
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_layer_2_layer_call_and_return_conditional_losses_24935711

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�/
�
#__inference__wrapped_model_24935275
input_2C
0criticnet_layer_1_matmul_readvariableop_resource:	�@
1criticnet_layer_1_biasadd_readvariableop_resource:	�D
0criticnet_layer_2_matmul_readvariableop_resource:
��@
1criticnet_layer_2_biasadd_readvariableop_resource:	�D
0criticnet_layer_3_matmul_readvariableop_resource:
��@
1criticnet_layer_3_biasadd_readvariableop_resource:	�A
.criticnet_dense_matmul_readvariableop_resource:	�=
/criticnet_dense_biasadd_readvariableop_resource:
identity��&CriticNet/dense/BiasAdd/ReadVariableOp�%CriticNet/dense/MatMul/ReadVariableOp�(CriticNet/layer_1/BiasAdd/ReadVariableOp�'CriticNet/layer_1/MatMul/ReadVariableOp�(CriticNet/layer_2/BiasAdd/ReadVariableOp�'CriticNet/layer_2/MatMul/ReadVariableOp�(CriticNet/layer_3/BiasAdd/ReadVariableOp�'CriticNet/layer_3/MatMul/ReadVariableOp�
'CriticNet/layer_1/MatMul/ReadVariableOpReadVariableOp0criticnet_layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02)
'CriticNet/layer_1/MatMul/ReadVariableOp�
CriticNet/layer_1/MatMulMatMulinput_2/CriticNet/layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
CriticNet/layer_1/MatMul�
(CriticNet/layer_1/BiasAdd/ReadVariableOpReadVariableOp1criticnet_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(CriticNet/layer_1/BiasAdd/ReadVariableOp�
CriticNet/layer_1/BiasAddBiasAdd"CriticNet/layer_1/MatMul:product:00CriticNet/layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
CriticNet/layer_1/BiasAdd�
CriticNet/layer_1/TanhTanh"CriticNet/layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
CriticNet/layer_1/Tanh�
'CriticNet/layer_2/MatMul/ReadVariableOpReadVariableOp0criticnet_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02)
'CriticNet/layer_2/MatMul/ReadVariableOp�
CriticNet/layer_2/MatMulMatMulCriticNet/layer_1/Tanh:y:0/CriticNet/layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
CriticNet/layer_2/MatMul�
(CriticNet/layer_2/BiasAdd/ReadVariableOpReadVariableOp1criticnet_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(CriticNet/layer_2/BiasAdd/ReadVariableOp�
CriticNet/layer_2/BiasAddBiasAdd"CriticNet/layer_2/MatMul:product:00CriticNet/layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
CriticNet/layer_2/BiasAdd�
CriticNet/layer_2/TanhTanh"CriticNet/layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
CriticNet/layer_2/Tanh�
'CriticNet/layer_3/MatMul/ReadVariableOpReadVariableOp0criticnet_layer_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02)
'CriticNet/layer_3/MatMul/ReadVariableOp�
CriticNet/layer_3/MatMulMatMulCriticNet/layer_2/Tanh:y:0/CriticNet/layer_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
CriticNet/layer_3/MatMul�
(CriticNet/layer_3/BiasAdd/ReadVariableOpReadVariableOp1criticnet_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(CriticNet/layer_3/BiasAdd/ReadVariableOp�
CriticNet/layer_3/BiasAddBiasAdd"CriticNet/layer_3/MatMul:product:00CriticNet/layer_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
CriticNet/layer_3/BiasAdd�
CriticNet/layer_3/TanhTanh"CriticNet/layer_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
CriticNet/layer_3/Tanh�
%CriticNet/dense/MatMul/ReadVariableOpReadVariableOp.criticnet_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02'
%CriticNet/dense/MatMul/ReadVariableOp�
CriticNet/dense/MatMulMatMulCriticNet/layer_3/Tanh:y:0-CriticNet/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
CriticNet/dense/MatMul�
&CriticNet/dense/BiasAdd/ReadVariableOpReadVariableOp/criticnet_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&CriticNet/dense/BiasAdd/ReadVariableOp�
CriticNet/dense/BiasAddBiasAdd CriticNet/dense/MatMul:product:0.CriticNet/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
CriticNet/dense/BiasAdd�
CriticNet/dense/SoftplusSoftplus CriticNet/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
CriticNet/dense/Softplus�
IdentityIdentity&CriticNet/dense/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp'^CriticNet/dense/BiasAdd/ReadVariableOp&^CriticNet/dense/MatMul/ReadVariableOp)^CriticNet/layer_1/BiasAdd/ReadVariableOp(^CriticNet/layer_1/MatMul/ReadVariableOp)^CriticNet/layer_2/BiasAdd/ReadVariableOp(^CriticNet/layer_2/MatMul/ReadVariableOp)^CriticNet/layer_3/BiasAdd/ReadVariableOp(^CriticNet/layer_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2P
&CriticNet/dense/BiasAdd/ReadVariableOp&CriticNet/dense/BiasAdd/ReadVariableOp2N
%CriticNet/dense/MatMul/ReadVariableOp%CriticNet/dense/MatMul/ReadVariableOp2T
(CriticNet/layer_1/BiasAdd/ReadVariableOp(CriticNet/layer_1/BiasAdd/ReadVariableOp2R
'CriticNet/layer_1/MatMul/ReadVariableOp'CriticNet/layer_1/MatMul/ReadVariableOp2T
(CriticNet/layer_2/BiasAdd/ReadVariableOp(CriticNet/layer_2/BiasAdd/ReadVariableOp2R
'CriticNet/layer_2/MatMul/ReadVariableOp'CriticNet/layer_2/MatMul/ReadVariableOp2T
(CriticNet/layer_3/BiasAdd/ReadVariableOp(CriticNet/layer_3/BiasAdd/ReadVariableOp2R
'CriticNet/layer_3/MatMul/ReadVariableOp'CriticNet/layer_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�'
�
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935606

inputs9
&layer_1_matmul_readvariableop_resource:	�6
'layer_1_biasadd_readvariableop_resource:	�:
&layer_2_matmul_readvariableop_resource:
��6
'layer_2_biasadd_readvariableop_resource:	�:
&layer_3_matmul_readvariableop_resource:
��6
'layer_3_biasadd_readvariableop_resource:	�7
$dense_matmul_readvariableop_resource:	�3
%dense_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�layer_1/BiasAdd/ReadVariableOp�layer_1/MatMul/ReadVariableOp�layer_2/BiasAdd/ReadVariableOp�layer_2/MatMul/ReadVariableOp�layer_3/BiasAdd/ReadVariableOp�layer_3/MatMul/ReadVariableOp�
layer_1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
layer_1/MatMul/ReadVariableOp�
layer_1/MatMulMatMulinputs%layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
layer_1/MatMul�
layer_1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
layer_1/BiasAdd/ReadVariableOp�
layer_1/BiasAddBiasAddlayer_1/MatMul:product:0&layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
layer_1/BiasAddq
layer_1/TanhTanhlayer_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
layer_1/Tanh�
layer_2/MatMul/ReadVariableOpReadVariableOp&layer_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
layer_2/MatMul/ReadVariableOp�
layer_2/MatMulMatMullayer_1/Tanh:y:0%layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
layer_2/MatMul�
layer_2/BiasAdd/ReadVariableOpReadVariableOp'layer_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
layer_2/BiasAdd/ReadVariableOp�
layer_2/BiasAddBiasAddlayer_2/MatMul:product:0&layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
layer_2/BiasAddq
layer_2/TanhTanhlayer_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
layer_2/Tanh�
layer_3/MatMul/ReadVariableOpReadVariableOp&layer_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
layer_3/MatMul/ReadVariableOp�
layer_3/MatMulMatMullayer_2/Tanh:y:0%layer_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
layer_3/MatMul�
layer_3/BiasAdd/ReadVariableOpReadVariableOp'layer_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
layer_3/BiasAdd/ReadVariableOp�
layer_3/BiasAddBiasAddlayer_3/MatMul:product:0&layer_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
layer_3/BiasAddq
layer_3/TanhTanhlayer_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
layer_3/Tanh�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMullayer_3/Tanh:y:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/BiasAddv
dense/SoftplusSoftplusdense/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense/Softplusw
IdentityIdentitydense/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^layer_1/BiasAdd/ReadVariableOp^layer_1/MatMul/ReadVariableOp^layer_2/BiasAdd/ReadVariableOp^layer_2/MatMul/ReadVariableOp^layer_3/BiasAdd/ReadVariableOp^layer_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
layer_1/BiasAdd/ReadVariableOplayer_1/BiasAdd/ReadVariableOp2>
layer_1/MatMul/ReadVariableOplayer_1/MatMul/ReadVariableOp2@
layer_2/BiasAdd/ReadVariableOplayer_2/BiasAdd/ReadVariableOp2>
layer_2/MatMul/ReadVariableOplayer_2/MatMul/ReadVariableOp2@
layer_3/BiasAdd/ReadVariableOplayer_3/BiasAdd/ReadVariableOp2>
layer_3/MatMul/ReadVariableOplayer_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935521
input_2#
layer_1_24935500:	�
layer_1_24935502:	�$
layer_2_24935505:
��
layer_2_24935507:	�$
layer_3_24935510:
��
layer_3_24935512:	�!
dense_24935515:	�
dense_24935517:
identity��dense/StatefulPartitionedCall�layer_1/StatefulPartitionedCall�layer_2/StatefulPartitionedCall�layer_3/StatefulPartitionedCall�
layer_1/StatefulPartitionedCallStatefulPartitionedCallinput_2layer_1_24935500layer_1_24935502*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_1_layer_call_and_return_conditional_losses_249352932!
layer_1/StatefulPartitionedCall�
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_24935505layer_2_24935507*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_2_layer_call_and_return_conditional_losses_249353102!
layer_2/StatefulPartitionedCall�
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_24935510layer_3_24935512*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_3_layer_call_and_return_conditional_losses_249353272!
layer_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall(layer_3/StatefulPartitionedCall:output:0dense_24935515dense_24935517*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_249353442
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
*__inference_layer_1_layer_call_fn_24935700

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_1_layer_call_and_return_conditional_losses_249352932
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935545
input_2#
layer_1_24935524:	�
layer_1_24935526:	�$
layer_2_24935529:
��
layer_2_24935531:	�$
layer_3_24935534:
��
layer_3_24935536:	�!
dense_24935539:	�
dense_24935541:
identity��dense/StatefulPartitionedCall�layer_1/StatefulPartitionedCall�layer_2/StatefulPartitionedCall�layer_3/StatefulPartitionedCall�
layer_1/StatefulPartitionedCallStatefulPartitionedCallinput_2layer_1_24935524layer_1_24935526*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_1_layer_call_and_return_conditional_losses_249352932!
layer_1/StatefulPartitionedCall�
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_24935529layer_2_24935531*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_2_layer_call_and_return_conditional_losses_249353102!
layer_2/StatefulPartitionedCall�
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_24935534layer_3_24935536*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_3_layer_call_and_return_conditional_losses_249353272!
layer_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall(layer_3/StatefulPartitionedCall:output:0dense_24935539dense_24935541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_249353442
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�	
�
&__inference_signature_wrapper_24935574
input_2
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*1
config_proto!

CPU

GPU 
(
2J 8� *,
f'R%
#__inference__wrapped_model_249352752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935351

inputs#
layer_1_24935294:	�
layer_1_24935296:	�$
layer_2_24935311:
��
layer_2_24935313:	�$
layer_3_24935328:
��
layer_3_24935330:	�!
dense_24935345:	�
dense_24935347:
identity��dense/StatefulPartitionedCall�layer_1/StatefulPartitionedCall�layer_2/StatefulPartitionedCall�layer_3/StatefulPartitionedCall�
layer_1/StatefulPartitionedCallStatefulPartitionedCallinputslayer_1_24935294layer_1_24935296*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_1_layer_call_and_return_conditional_losses_249352932!
layer_1/StatefulPartitionedCall�
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_24935311layer_2_24935313*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_2_layer_call_and_return_conditional_losses_249353102!
layer_2/StatefulPartitionedCall�
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_24935328layer_3_24935330*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_3_layer_call_and_return_conditional_losses_249353272!
layer_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall(layer_3/StatefulPartitionedCall:output:0dense_24935345dense_24935347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_249353442
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_layer_2_layer_call_fn_24935720

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_2_layer_call_and_return_conditional_losses_249353102
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_layer_1_layer_call_and_return_conditional_losses_24935691

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_layer_2_layer_call_and_return_conditional_losses_24935310

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
,__inference_CriticNet_layer_call_fn_24935659

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*1
config_proto!

CPU

GPU 
(
2J 8� *P
fKRI
G__inference_CriticNet_layer_call_and_return_conditional_losses_249353512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_layer_3_layer_call_fn_24935740

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *N
fIRG
E__inference_layer_3_layer_call_and_return_conditional_losses_249353272
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
͆
�
$__inference__traced_restore_24935979
file_prefix2
assignvariableop_layer_1_kernel:	�.
assignvariableop_1_layer_1_bias:	�5
!assignvariableop_2_layer_2_kernel:
��.
assignvariableop_3_layer_2_bias:	�5
!assignvariableop_4_layer_3_kernel:
��.
assignvariableop_5_layer_3_bias:	�2
assignvariableop_6_dense_kernel:	�+
assignvariableop_7_dense_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: <
)assignvariableop_15_adam_layer_1_kernel_m:	�6
'assignvariableop_16_adam_layer_1_bias_m:	�=
)assignvariableop_17_adam_layer_2_kernel_m:
��6
'assignvariableop_18_adam_layer_2_bias_m:	�=
)assignvariableop_19_adam_layer_3_kernel_m:
��6
'assignvariableop_20_adam_layer_3_bias_m:	�:
'assignvariableop_21_adam_dense_kernel_m:	�3
%assignvariableop_22_adam_dense_bias_m:<
)assignvariableop_23_adam_layer_1_kernel_v:	�6
'assignvariableop_24_adam_layer_1_bias_v:	�=
)assignvariableop_25_adam_layer_2_kernel_v:
��6
'assignvariableop_26_adam_layer_2_bias_v:	�=
)assignvariableop_27_adam_layer_3_kernel_v:
��6
'assignvariableop_28_adam_layer_3_bias_v:	�:
'assignvariableop_29_adam_dense_kernel_v:	�3
%assignvariableop_30_adam_dense_bias_v:
identity_32��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_layer_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_layer_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_layer_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_layer_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_layer_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_layer_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_layer_2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_layer_3_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_layer_3_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_layer_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_layer_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_layer_2_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_layer_2_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_layer_3_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_layer_3_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_dense_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31f
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_32�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
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
�
�
E__inference_layer_3_layer_call_and_return_conditional_losses_24935731

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�'
�
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935638

inputs9
&layer_1_matmul_readvariableop_resource:	�6
'layer_1_biasadd_readvariableop_resource:	�:
&layer_2_matmul_readvariableop_resource:
��6
'layer_2_biasadd_readvariableop_resource:	�:
&layer_3_matmul_readvariableop_resource:
��6
'layer_3_biasadd_readvariableop_resource:	�7
$dense_matmul_readvariableop_resource:	�3
%dense_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�layer_1/BiasAdd/ReadVariableOp�layer_1/MatMul/ReadVariableOp�layer_2/BiasAdd/ReadVariableOp�layer_2/MatMul/ReadVariableOp�layer_3/BiasAdd/ReadVariableOp�layer_3/MatMul/ReadVariableOp�
layer_1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
layer_1/MatMul/ReadVariableOp�
layer_1/MatMulMatMulinputs%layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
layer_1/MatMul�
layer_1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
layer_1/BiasAdd/ReadVariableOp�
layer_1/BiasAddBiasAddlayer_1/MatMul:product:0&layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
layer_1/BiasAddq
layer_1/TanhTanhlayer_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
layer_1/Tanh�
layer_2/MatMul/ReadVariableOpReadVariableOp&layer_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
layer_2/MatMul/ReadVariableOp�
layer_2/MatMulMatMullayer_1/Tanh:y:0%layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
layer_2/MatMul�
layer_2/BiasAdd/ReadVariableOpReadVariableOp'layer_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
layer_2/BiasAdd/ReadVariableOp�
layer_2/BiasAddBiasAddlayer_2/MatMul:product:0&layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
layer_2/BiasAddq
layer_2/TanhTanhlayer_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
layer_2/Tanh�
layer_3/MatMul/ReadVariableOpReadVariableOp&layer_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
layer_3/MatMul/ReadVariableOp�
layer_3/MatMulMatMullayer_2/Tanh:y:0%layer_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
layer_3/MatMul�
layer_3/BiasAdd/ReadVariableOpReadVariableOp'layer_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
layer_3/BiasAdd/ReadVariableOp�
layer_3/BiasAddBiasAddlayer_3/MatMul:product:0&layer_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
layer_3/BiasAddq
layer_3/TanhTanhlayer_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
layer_3/Tanh�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMullayer_3/Tanh:y:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/BiasAddv
dense/SoftplusSoftplusdense/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense/Softplusw
IdentityIdentitydense/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^layer_1/BiasAdd/ReadVariableOp^layer_1/MatMul/ReadVariableOp^layer_2/BiasAdd/ReadVariableOp^layer_2/MatMul/ReadVariableOp^layer_3/BiasAdd/ReadVariableOp^layer_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
layer_1/BiasAdd/ReadVariableOplayer_1/BiasAdd/ReadVariableOp2>
layer_1/MatMul/ReadVariableOplayer_1/MatMul/ReadVariableOp2@
layer_2/BiasAdd/ReadVariableOplayer_2/BiasAdd/ReadVariableOp2>
layer_2/MatMul/ReadVariableOplayer_2/MatMul/ReadVariableOp2@
layer_3/BiasAdd/ReadVariableOplayer_3/BiasAdd/ReadVariableOp2>
layer_3/MatMul/ReadVariableOplayer_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�E
�
!__inference__traced_save_24935876
file_prefix-
)savev2_layer_1_kernel_read_readvariableop+
'savev2_layer_1_bias_read_readvariableop-
)savev2_layer_2_kernel_read_readvariableop+
'savev2_layer_2_bias_read_readvariableop-
)savev2_layer_3_kernel_read_readvariableop+
'savev2_layer_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_layer_1_kernel_m_read_readvariableop2
.savev2_adam_layer_1_bias_m_read_readvariableop4
0savev2_adam_layer_2_kernel_m_read_readvariableop2
.savev2_adam_layer_2_bias_m_read_readvariableop4
0savev2_adam_layer_3_kernel_m_read_readvariableop2
.savev2_adam_layer_3_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_layer_1_kernel_v_read_readvariableop2
.savev2_adam_layer_1_bias_v_read_readvariableop4
0savev2_adam_layer_2_kernel_v_read_readvariableop2
.savev2_adam_layer_2_bias_v_read_readvariableop4
0savev2_adam_layer_3_kernel_v_read_readvariableop2
.savev2_adam_layer_3_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_layer_1_kernel_read_readvariableop'savev2_layer_1_bias_read_readvariableop)savev2_layer_2_kernel_read_readvariableop'savev2_layer_2_bias_read_readvariableop)savev2_layer_3_kernel_read_readvariableop'savev2_layer_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_layer_1_kernel_m_read_readvariableop.savev2_adam_layer_1_bias_m_read_readvariableop0savev2_adam_layer_2_kernel_m_read_readvariableop.savev2_adam_layer_2_bias_m_read_readvariableop0savev2_adam_layer_3_kernel_m_read_readvariableop.savev2_adam_layer_3_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_layer_1_kernel_v_read_readvariableop.savev2_adam_layer_1_bias_v_read_readvariableop0savev2_adam_layer_2_kernel_v_read_readvariableop.savev2_adam_layer_2_bias_v_read_readvariableop0savev2_adam_layer_3_kernel_v_read_readvariableop.savev2_adam_layer_3_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�:�:
��:�:
��:�:	�:: : : : : : : :	�:�:
��:�:
��:�:	�::	�:�:
��:�:
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
:: 

_output_shapes
: 
�
�
C__inference_dense_layer_call_and_return_conditional_losses_24935344

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_layer_call_fn_24935760

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU 
(
2J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_249353442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
,__inference_CriticNet_layer_call_fn_24935370
input_2
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*1
config_proto!

CPU

GPU 
(
2J 8� *P
fKRI
G__inference_CriticNet_layer_call_and_return_conditional_losses_249353512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�

�
,__inference_CriticNet_layer_call_fn_24935497
input_2
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*1
config_proto!

CPU

GPU 
(
2J 8� *P
fKRI
G__inference_CriticNet_layer_call_and_return_conditional_losses_249354572
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
C__inference_dense_layer_call_and_return_conditional_losses_24935751

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
,__inference_CriticNet_layer_call_fn_24935680

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*1
config_proto!

CPU

GPU 
(
2J 8� *P
fKRI
G__inference_CriticNet_layer_call_and_return_conditional_losses_249354572
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_layer_3_layer_call_and_return_conditional_losses_24935327

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_layer_1_layer_call_and_return_conditional_losses_24935293

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanhd
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_20
serving_default_input_2:0���������9
dense0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�_
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer

signatures
#_self_saveable_object_factories
	regularization_losses

trainable_variables
	variables
	keras_api
*^&call_and_return_all_conditional_losses
___call__
`_default_save_signature"
_tf_keras_network
[
_init_input_shape
#_self_saveable_object_factories"
_tf_keras_input_layer
�

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"
_tf_keras_layer
�

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"
_tf_keras_layer
�

kernel
bias
#_self_saveable_object_factories
 regularization_losses
!trainable_variables
"	variables
#	keras_api
*e&call_and_return_all_conditional_losses
f__call__"
_tf_keras_layer
�

$kernel
%bias
#&_self_saveable_object_factories
'regularization_losses
(trainable_variables
)	variables
*	keras_api
*g&call_and_return_all_conditional_losses
h__call__"
_tf_keras_layer
�
+iter

,beta_1

-beta_2
	.decay
/learning_ratemNmOmPmQmRmS$mT%mUvVvWvXvYvZv[$v\%v]"
	optimizer
,
iserving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
$6
%7"
trackable_list_wrapper
X
0
1
2
3
4
5
$6
%7"
trackable_list_wrapper
�
	regularization_losses
0layer_metrics

1layers
2non_trainable_variables

trainable_variables
3metrics
4layer_regularization_losses
	variables
___call__
`_default_save_signature
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
!:	�2layer_1/kernel
:�2layer_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
5layer_metrics

6layers
7metrics
8non_trainable_variables
trainable_variables
9layer_regularization_losses
	variables
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
": 
��2layer_2/kernel
:�2layer_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
:layer_metrics

;layers
<metrics
=non_trainable_variables
trainable_variables
>layer_regularization_losses
	variables
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
": 
��2layer_3/kernel
:�2layer_3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
 regularization_losses
?layer_metrics

@layers
Ametrics
Bnon_trainable_variables
!trainable_variables
Clayer_regularization_losses
"	variables
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
:	�2dense/kernel
:2
dense/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
�
'regularization_losses
Dlayer_metrics

Elayers
Fmetrics
Gnon_trainable_variables
(trainable_variables
Hlayer_regularization_losses
)	variables
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
'
I0"
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
N
	Jtotal
	Kcount
L	variables
M	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
J0
K1"
trackable_list_wrapper
-
L	variables"
_generic_user_object
&:$	�2Adam/layer_1/kernel/m
 :�2Adam/layer_1/bias/m
':%
��2Adam/layer_2/kernel/m
 :�2Adam/layer_2/bias/m
':%
��2Adam/layer_3/kernel/m
 :�2Adam/layer_3/bias/m
$:"	�2Adam/dense/kernel/m
:2Adam/dense/bias/m
&:$	�2Adam/layer_1/kernel/v
 :�2Adam/layer_1/bias/v
':%
��2Adam/layer_2/kernel/v
 :�2Adam/layer_2/bias/v
':%
��2Adam/layer_3/kernel/v
 :�2Adam/layer_3/bias/v
$:"	�2Adam/dense/kernel/v
:2Adam/dense/bias/v
�2�
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935606
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935638
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935521
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935545�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_CriticNet_layer_call_fn_24935370
,__inference_CriticNet_layer_call_fn_24935659
,__inference_CriticNet_layer_call_fn_24935680
,__inference_CriticNet_layer_call_fn_24935497�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
#__inference__wrapped_model_24935275input_2"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_layer_1_layer_call_and_return_conditional_losses_24935691�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_layer_1_layer_call_fn_24935700�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_layer_2_layer_call_and_return_conditional_losses_24935711�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_layer_2_layer_call_fn_24935720�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_layer_3_layer_call_and_return_conditional_losses_24935731�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_layer_3_layer_call_fn_24935740�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_layer_call_and_return_conditional_losses_24935751�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_layer_call_fn_24935760�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_24935574input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935521k$%8�5
.�+
!�
input_2���������
p 

 
� "%�"
�
0���������
� �
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935545k$%8�5
.�+
!�
input_2���������
p

 
� "%�"
�
0���������
� �
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935606j$%7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
G__inference_CriticNet_layer_call_and_return_conditional_losses_24935638j$%7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
,__inference_CriticNet_layer_call_fn_24935370^$%8�5
.�+
!�
input_2���������
p 

 
� "�����������
,__inference_CriticNet_layer_call_fn_24935497^$%8�5
.�+
!�
input_2���������
p

 
� "�����������
,__inference_CriticNet_layer_call_fn_24935659]$%7�4
-�*
 �
inputs���������
p 

 
� "�����������
,__inference_CriticNet_layer_call_fn_24935680]$%7�4
-�*
 �
inputs���������
p

 
� "�����������
#__inference__wrapped_model_24935275k$%0�-
&�#
!�
input_2���������
� "-�*
(
dense�
dense����������
C__inference_dense_layer_call_and_return_conditional_losses_24935751]$%0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� |
(__inference_dense_layer_call_fn_24935760P$%0�-
&�#
!�
inputs����������
� "�����������
E__inference_layer_1_layer_call_and_return_conditional_losses_24935691]/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� ~
*__inference_layer_1_layer_call_fn_24935700P/�,
%�"
 �
inputs���������
� "������������
E__inference_layer_2_layer_call_and_return_conditional_losses_24935711^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_layer_2_layer_call_fn_24935720Q0�-
&�#
!�
inputs����������
� "������������
E__inference_layer_3_layer_call_and_return_conditional_losses_24935731^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_layer_3_layer_call_fn_24935740Q0�-
&�#
!�
inputs����������
� "������������
&__inference_signature_wrapper_24935574v$%;�8
� 
1�.
,
input_2!�
input_2���������"-�*
(
dense�
dense���������