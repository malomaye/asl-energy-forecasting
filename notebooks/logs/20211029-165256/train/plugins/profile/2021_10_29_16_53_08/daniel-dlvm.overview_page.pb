?	?N???=@?N???=@!?N???=@	~??a???~??a???!~??a???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?N???=@(,????@A?t???:;@Y????Z???*	?rh?) A2?
lIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2[0]::CSVE#k??(O@!8??G?G@)#k??(O@18??G?G@:Preprocessing2?
dIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2B#?W<?:L@!Ҩ?wʂE@)#?W<?:L@1Ҩ?wʂE@:Preprocessing2m
6Iterator::Model::Prefetch::Map::Prefetch::Map::BatchV2Z?b+h?P@!?{?N?I@)_?L?:#@1?h???M@:Preprocessing2
HIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeatB? ?w?6M@!???d?BF@)?,?i???1q??= ??:Preprocessing2U
Iterator::Model::Prefetch::Map
pU???!Si?"?Y??)??.5B???1?&?w???:Preprocessing2d
-Iterator::Model::Prefetch::Map::Prefetch::Map?4?\?P@!??z?I@)#e??????1?0?iW??:Preprocessing2_
(Iterator::Model::Prefetch::Map::Prefetch??
??X??!??YEo??)??
??X??1??YEo??:Preprocessing2F
Iterator::Model??8~??!?+?Aǩ??)?k	??g??1*??2????:Preprocessing2P
Iterator::Model::Prefetch????Д??!p?Q?q??)????Д??1p?Q?q??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9~??a???>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	(,????@(,????@!(,????@      ??!       "      ??!       *      ??!       2	?t???:;@?t???:;@!?t???:;@:      ??!       B      ??!       J	????Z???????Z???!????Z???R      ??!       Z	????Z???????Z???!????Z???JCPU_ONLYY~??a???b Y      Y@q*Z?U;@"?
both?Your program is POTENTIALLY input-bound because 7.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 