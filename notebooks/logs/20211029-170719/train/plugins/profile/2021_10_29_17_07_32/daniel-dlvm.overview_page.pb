?	?FY???3@?FY???3@!?FY???3@	?}?????}????!?}????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?FY???3@?RE????A?ǵ?b?2@Y4w??o??*	W-?E??@2?
lIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2[0]::CSV L?$z?)@!H?r*E@)L?$z?)@1H?r*E@:Preprocessing2?
dIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2 ???%?g%@!?P???A@)???%?g%@1?P???A@:Preprocessing2m
6Iterator::Model::Prefetch::Map::Prefetch::Map::BatchV2mr??A0@!?v???J@)?q????@1??1"?.@:Preprocessing2U
Iterator::Model::Prefetch::Map?Un2???!ʀN??@)?G???1 ?:4?"@:Preprocessing2
HIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat ?+?S'@!=?-??B@)?DeÚ???17þ}?@:Preprocessing2_
(Iterator::Model::Prefetch::Map::PrefetchND??~??!?<1?W??)ND??~??1?<1?W??:Preprocessing2d
-Iterator::Model::Prefetch::Map::Prefetch::Map}????Q0@!]?u???J@)(v?U???13J??C??:Preprocessing2F
Iterator::Model????[??!HN?????)?*3?????1?9?????:Preprocessing2P
Iterator::Model::Prefetch????O???!)?牢??)????O???1)?牢??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?}????>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?RE?????RE????!?RE????      ??!       "      ??!       *      ??!       2	?ǵ?b?2@?ǵ?b?2@!?ǵ?b?2@:      ??!       B      ??!       J	4w??o??4w??o??!4w??o??R      ??!       Z	4w??o??4w??o??!4w??o??JCPU_ONLYY?}????b Y      Y@q~??W????"?
both?Your program is POTENTIALLY input-bound because 3.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 