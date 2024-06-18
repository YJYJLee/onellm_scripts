export LD_LIBRARY_PATH=/fsx-atom/yejinlee/condaenv/seamless/lib:$LD_LIBRARY_PATH

m4t_evaluate --data_file /fsx-atom/yejinlee/seamless_dataset2/test_manifest.json --task S2ST --tgt_lang spa --output_path ./test --ref_field tgt_text --audio_root_dir /fsx-atom/yejinlee/seamless_dataset2/downloads/extracted/7411e52500e75f4587cdf715b30663e27f8e5f4865174435569bffe9b3899945/test --model_name seamlessM4T_v2_large --batch_size 1 > /fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/S2ST/log.txt

m4t_evaluate --data_file /fsx-atom/yejinlee/seamless_dataset2/test_manifest.json --task S2TT --tgt_lang spa --output_path ./test --ref_field tgt_text --audio_root_dir /fsx-atom/yejinlee/seamless_dataset2/downloads/extracted/7411e52500e75f4587cdf715b30663e27f8e5f4865174435569bffe9b3899945/test --model_name seamlessM4T_v2_large --batch_size 1 > /fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/S2TT/l
og.txt

m4t_evaluate --data_file /fsx-atom/yejinlee/seamless_dataset2/test_manifest.json --task T2TT --tgt_lang spa --output_path ./test --ref_field tgt_text --audio_root_dir /fsx-atom/yejinlee/seamless_dataset2/downloads/extracted/7411e52500e75f4587cdf715b30663e27f8e5f4865174435569bffe9b3899945/test --model_name seamlessM4T_v2_large --batch_size 1 > /fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/T2TT/log.txt

m4t_evaluate --data_file /fsx-atom/yejinlee/seamless_dataset2/test_manifest.json --task T2ST --tgt_lang spa --output_path ./test --ref_field tgt_text --audio_root_dir /fsx-atom/yejinlee/seamless_dataset2/downloads/extracted/7411e52500e75f4587cdf715b30663e27f8e5f4865174435569bffe9b3899945/test --model_name seamlessM4T_v2_large --batch_size 1 > /fsx-atom/yejinlee/paper_submission_results/sequence_lengths/seamless/fleurs/T2ST/log.txt
