# Standard (16-bit, strict <8e-6 error)
cpp\build\dlc.exe compress -i test_data\sine_1m.bin -o output.dlc --precision 16

# Aggressive compression (12-bit, ~1.2e-4 error, higher ratio)
cpp\build\dlc.exe compress -i test_data\sine_1m.bin -o output.dlc --precision 12

# Near-lossless (20-bit, tightest error, lower ratio)
cpp\build\dlc.exe compress -i test_data\sine_1m.bin -o output.dlc --precision 20

# Maximum compression (8-bit, lossy)
cpp\build\dlc.exe compress -i test_data\sine_1m.bin -o output.dlc --precision 8

# Compress a File
cpp\build\dlc.exe compress -i test_data\massive_sensor_3m.bin -o test_data\my_compressed_file.dlc --workers 4

# Decompress a file
cpp\build\dlc.exe decompress -i test_data\my_compressed_file.dlc -o test_data\reconstructed_sensor.bin

# Run the Dashboard
streamlit run python/app.py

# Benchmarks & Visualization
# Generate all test datasets
python test_data/generate_test_data.py
python test_data/download_massive_data.py

# Terminal comparison across ALL datasets
python python/benchmarks/compare_all.py

# Generate presentation charts
python python/benchmarks/visualize_real_data.py
python python/benchmarks/visualize_final_dashboards.py

# To generate the full 700MB+ datasets, just run:
python test_data/download_large_datasets.py

# Dataset
python test_data/generate_test_data.py
python test_data/download_massive_data.py