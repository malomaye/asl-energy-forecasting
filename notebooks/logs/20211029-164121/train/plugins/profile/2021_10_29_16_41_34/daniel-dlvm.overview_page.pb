?	??<????@??<????@!??<????@	Ѡ??O??Ѡ??O??!Ѡ??O??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??<????@h˹W??A?πz3n>@Y)??????*	h??|???@2?
lIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2[0]::CSV (?4?t,@!???H?D@)(?4?t,@1???H?D@:Preprocessing2?
dIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2 R????(@! ?3??A@)R????(@1 ?3??A@:Preprocessing2m
6Iterator::Model::Prefetch::Map::Prefetch::Map::BatchV2k???t?1@!gį??I@)?UIdd@1ۆv???*@:Preprocessing2U
Iterator::Model::Prefetch::Map?1?Mc???!??,)@)??g??s??1??2??-@:Preprocessing2
HIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat ?9??x*@!f?f?~GC@)?h>????1SV/???@:Preprocessing2F
Iterator::Model??9???!?????o??)[
H?`??1j]?"??:Preprocessing2_
(Iterator::Model::Prefetch::Map::Prefetch=Y??w??!I?%:????)=Y??w??1I?%:????:Preprocessing2d
-Iterator::Model::Prefetch::Map::Prefetch::Map?B;?Y?1@!q???rJ@)AaP?????13Tb50???:Preprocessing2P
Iterator::Model::Prefetchf2?g@??!?+?t[???)f2?g@??1?+?t[???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Ѡ??O??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	h˹W??h˹W??!h˹W??      ??!       "      ??!       *      ??!       2	?πz3n>@?πz3n>@!?πz3n>@:      ??!       B      ??!       J	)??????)??????!)??????R      ??!       Z	)??????)??????!)??????JCPU_ONLYYѠ??O??b Y      Y@qBu?{=??"?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 