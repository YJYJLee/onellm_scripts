export LD_LIBRARY_PATH=/fsx-atom/yejinlee/condaenv/seamless/lib:$LD_LIBRARY_PATH

BATCH_SIZE=$1
m4t_evaluate --data_file /fsx-atom/shared/yejinlee_backups/seamless_dataset2/test_manifest.json --task S2ST --tgt_lang spa --output_path ./test --ref_field tgt_text --audio_root_dir /fsx-atom/shared/yejinlee_backups/seamless_dataset2/downloads/extracted/7411e52500e75f4587cdf715b30663e27f8e5f4865174435569bffe9b3899945/test --model_name seamlessM4T_v2_large --batch_size $BATCH_SIZE

m4t_evaluate --data_file /fsx-atom/shared/yejinlee_backups/seamless_dataset2/test_manifest.json --task S2TT --tgt_lang spa --output_path ./test --ref_field tgt_text --audio_root_dir /fsx-atom/shared/yejinlee_backups/seamless_dataset2/downloads/extracted/7411e52500e75f4587cdf715b30663e27f8e5f4865174435569bffe9b3899945/test --model_name seamlessM4T_v2_large --batch_size $BATCH_SIZE

m4t_evaluate --data_file /fsx-atom/shared/yejinlee_backups/seamless_dataset2/test_manifest.json --task T2TT --tgt_lang spa --output_path ./test --ref_field tgt_text --audio_root_dir /fsx-atom/shared/yejinlee_backups/seamless_dataset2/downloads/extracted/7411e52500e75f4587cdf715b30663e27f8e5f4865174435569bffe9b3899945/test --model_name seamlessM4T_v2_large --batch_size $BATCH_SIZE

m4t_evaluate --data_file /fsx-atom/shared/yejinlee_backups/seamless_dataset2/test_manifest.json --task T2ST --tgt_lang spa --output_path ./test --ref_field tgt_text --audio_root_dir /fsx-atom/shared/yejinlee_backups/seamless_dataset2/downloads/extracted/7411e52500e75f4587cdf715b30663e27f8e5f4865174435569bffe9b3899945/test --model_name seamlessM4T_v2_large --batch_size $BATCH_SIZE
