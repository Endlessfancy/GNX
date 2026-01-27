(MIX) PS C:\Private\Research\GNX_final\GNX\analysis> python .\compile_time_measure.py --repeat 10
Loading profiling data...
Data loading: 3.97 ms
Initializing scheduler...
Scheduler init: 14.81 ms

======================================================================
K    Subgraphs    Opt Time (ms)      Makespan (ms)
======================================================================
Traceback (most recent call last):
  File "C:\Private\Research\GNX_final\GNX\analysis\compile_time_measure.py", line 259, in <module>
    main()
  File "C:\Private\Research\GNX_final\GNX\analysis\compile_time_measure.py", line 202, in main
    partitions = parse_flickr_partition(partition_file, k)
  File "C:\Private\Research\GNX_final\GNX\analysis\compile_time_measure.py", line 46, in parse_flickr_partition
    content = f.read()
  File "C:\Env\Anaconda\envs\MIX\lib\encodings\cp1252.py", line 23, in decode
    return codecs.charmap_decode(input,self.errors,decoding_table)[0]
UnicodeDecodeError: 'charmap' codec can't decode byte 0x8d in position 13: character maps to <undefined>