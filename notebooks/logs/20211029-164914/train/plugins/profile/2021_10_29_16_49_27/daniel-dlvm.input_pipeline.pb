	?lV}^:@?lV}^:@!?lV}^:@	?`??(????`??(???!?`??(???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?lV}^:@&?B????A?q75$9@Y?%?????*	֣p=^??@2?
dIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2 ????G*@!2?@+?-C@)????G*@12?@+?-C@:Preprocessing2?
lIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2[0]::CSV "ĕ?w?)@!	???B@)"ĕ?w?)@1	???B@:Preprocessing2m
6Iterator::Model::Prefetch::Map::Prefetch::Map::BatchV2???W;*3@!?r[??K@)w?n???@1p-fl[.@:Preprocessing2U
Iterator::Model::Prefetch::Map??A?p @!? ?Qf@)?QcB?%??14zN??@:Preprocessing2
HIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat ?Z?Q?+@!-???aD@)?z?Ga??1???H@@:Preprocessing2_
(Iterator::Model::Prefetch::Map::Prefetch???N???!??z?e??)???N???1??z?e??:Preprocessing2d
-Iterator::Model::Prefetch::Map::Prefetch::Map?˸?=3@!Z??L@)??an??1Xp???[??:Preprocessing2F
Iterator::Model?J̳?V??!?6????)l?衶??1}??5֮??:Preprocessing2P
Iterator::Model::Prefetch"?{????!?? ???)"?{????1?? ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?`??(???>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	&?B????&?B????!&?B????      ??!       "      ??!       *      ??!       2	?q75$9@?q75$9@!?q75$9@:      ??!       B      ??!       J	?%??????%?????!?%?????R      ??!       Z	?%??????%?????!?%?????JCPU_ONLYY?`??(???b 