	?f+/?C1@?f+/?C1@!?f+/?C1@	8?}[???8?}[???!8?}[???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?f+/?C1@?)?????A?>?x0@Y??{,G??*	.???7??@2?
lIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2[0]::CSV ??5?(@!??p]nC@)??5?(@1??p]nC@:Preprocessing2?
dIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2 ??OU??$@!??%??b@@)??OU??$@1??%??b@@:Preprocessing2m
6Iterator::Model::Prefetch::Map::Prefetch::Map::BatchV2n?@?
1@!?Y??K@)?%?<9@1?Ŷؔj2@:Preprocessing2U
Iterator::Model::Prefetch::MapB???8a??!˛v[?@)???]???1-???Ю@:Preprocessing2
HIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat ?Ac&y&@!??2?c?A@)???P???1a?? R?@:Preprocessing2F
Iterator::Model?%z????!?U?)p7??)W?sD?K??1?#?s???:Preprocessing2_
(Iterator::Model::Prefetch::Map::Prefetch??@?m??!??ޗf???)??@?m??1??ޗf???:Preprocessing2d
-Iterator::Model::Prefetch::Map::Prefetch::Map?? w1@!??y?,&K@)=_?\6:??1??H??~??:Preprocessing2P
Iterator::Model::Prefetch'??b??!M?d??G??)'??b??1M?d??G??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no99?}[???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?)??????)?????!?)?????      ??!       "      ??!       *      ??!       2	?>?x0@?>?x0@!?>?x0@:      ??!       B      ??!       J	??{,G????{,G??!??{,G??R      ??!       Z	??{,G????{,G??!??{,G??JCPU_ONLYY9?}[???b 